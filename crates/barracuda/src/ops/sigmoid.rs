// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sigmoid — GPU-resident, pipeline-cached, bind-group-cached, batchable
//!
//! Deep Debt Principles:
//! - Zero hardcoding: Capability-based workgroup dispatch
//! - Batchable: routes through `TensorContext::record_operation()`
//! - Zero-copy output: buffer pool, no GPU→CPU→GPU round-trip
//! - Pipeline cached: `GLOBAL_CACHE` eliminates recompilation overhead
//! - Bind-group cached: `get_or_create_bind_group()` reuses BG for same tensor pair

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/misc/sigmoid_f64.wgsl");
pub(crate) static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
});

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Sigmoid: `1 / (1 + e⁻ˣ)`
pub struct Sigmoid {
    input: Tensor,
}

impl Sigmoid {
    /// Creates a Sigmoid operation for the given input tensor.
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute Sigmoid.
    /// - Output stays GPU-resident (no readback).
    /// - Pipeline and bind group compiled/built once and cached globally.
    /// - Dispatch batched when inside `TensorSession`.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        let ctx = get_device_context(device);
        let caps = DeviceCapabilities::from_device(device);
        let wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let workgroups = (size as u32).div_ceil(wg_size);
        let adapter_info = device.adapter_info();

        // Pooled output — zero allocation in steady state.
        let output_buffer = ctx.acquire_pooled_output(size);

        // elementwise_unary: 1 read-only storage + 1 read-write storage (no uniform).
        // The bind group is fully determined by the buffer IDs → cached hit after first call.
        let layout_sig = BindGroupLayoutSignature::elementwise_unary();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[self.input.buffer(), &output_buffer],
            Some("Sigmoid BG"),
        );

        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            Self::wgsl_shader(),
            layout_sig,
            "main",
            Some("Sigmoid Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sigmoid Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        })?;

        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute Sigmoid element-wise (GPU-resident, pipeline-cached, batchable).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn sigmoid(self) -> Result<Self> {
        Sigmoid::new(self).execute()
    }
}
