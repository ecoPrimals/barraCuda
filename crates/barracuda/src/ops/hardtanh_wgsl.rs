// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardtanh — GPU-resident, pipeline-cached, batchable
//!
//! Deep Debt Principles:
//! - Zero hardcoding: Capability-based workgroup dispatch
//! - Batchable: routes through `TensorContext::record_operation()`
//! - Zero-copy output: buffer pool, no GPU→CPU→GPU round-trip
//! - Pipeline cached: `GLOBAL_CACHE` eliminates recompilation overhead

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

/// f64 is the canonical source.
const SHADER_F64: &str = include_str!("../shaders/activation/hardtanh_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    min_val: f32,
    max_val: f32,
}

/// Hardtanh activation.
pub struct Hardtanh {
    input: Tensor,
}

impl Hardtanh {
    /// Creates a Hardtanh operation for the given input tensor.
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute Hardtanh.
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

        let output_buffer = ctx.acquire_pooled_output(size);

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Hardtanh Params"),
                contents: bytemuck::bytes_of(&Params {
                    size: size as u32,
                    min_val: -1.0,
                    max_val: 1.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout_sig = BindGroupLayoutSignature::reduction();
        let adapter_info = device.adapter_info();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("Hardtanh BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Hardtanh BG"),
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
            }));

        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            Self::wgsl_shader(),
            layout_sig,
            "main",
            Some("Hardtanh Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hardtanh Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(params_buf);
        })?;

        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute Hardtanh element-wise (GPU-resident, pipeline-cached, batchable).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn hardtanh_wgsl(self) -> Result<Self> {
        Hardtanh::new(self).execute()
    }
}
