// SPDX-License-Identifier: AGPL-3.0-or-later
//! GELU — GPU-resident, pipeline-cached, batchable
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Capability-based workgroup dispatch
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: No mocks or stubs
//! - Batchable: routes through `TensorContext::record_operation()`
//! - Zero-copy output: buffer pool, no GPU→CPU→GPU round-trip
//! - Pipeline cached: `GLOBAL_CACHE` eliminates recompilation overhead

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/activation/gelu_f64.wgsl");
pub(crate) static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    SHADER_F64.to_string()
});

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
}

/// GELU activation: `x · Φ(x)` where Φ is the Gaussian CDF.
pub struct GELU {
    input: Tensor,
}

impl GELU {
    /// Creates a GELU operation for the given input tensor.
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute GELU.
    ///
    /// - Output stays GPU-resident (no readback).
    /// - Pipeline compiled once, cached globally.
    /// - Dispatch batched when inside `TensorSession`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        let ctx = get_device_context(device);
        let caps = DeviceCapabilities::from_device(device);
        let wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let workgroups = (size as u32).div_ceil(wg_size);

        // Pooled output — zero allocation in steady state.
        let output_buffer = ctx.acquire_pooled_output(size);

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GELU Params"),
                contents: bytemuck::bytes_of(&Params { size: size as u32 }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Cached BGL: reduction() = (1 read-only, 1 read-write, 1 uniform)
        let layout_sig = BindGroupLayoutSignature::reduction();
        let adapter_info = device.adapter_info();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("GELU BGL"),
        );

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GELU BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let bind_group = std::sync::Arc::new(bind_group);

        // Cached pipeline.
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            Self::wgsl_shader(),
            layout_sig,
            "main",
            Some("GELU Pipeline"),
        );

        // Record into TensorContext — immediate when not batching, deferred otherwise.
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GELU Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(params_buf); // keep alive until dispatch
        })?;

        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute GELU element-wise (GPU-resident, pipeline-cached, batchable).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn gelu_wgsl(self) -> Result<Self> {
        GELU::new(self).execute()
    }
}
