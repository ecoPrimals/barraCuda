// SPDX-License-Identifier: AGPL-3.0-or-later
//! ELU — GPU-resident, pipeline-cached, batchable
//!
//! Deep Debt Principles:
//! - Zero hardcoding: Capability-based workgroup dispatch
//! - Batchable: routes through `TensorContext::record_operation()`
//! - Zero-copy output: buffer pool, no GPU→CPU→GPU round-trip
//! - Pipeline cached: `GLOBAL_CACHE` eliminates recompilation overhead
//! - Params fixed (S14): Rust `Params` matches WGSL `{ size, alpha }`

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_ELU_F64: &str = include_str!("../shaders/activation/elu_f64.wgsl");
const SHADER_ELU_SIMPLE_F64: &str = include_str!("../shaders/activation/elu_simple_f64.wgsl");
pub(crate) static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    SHADER_ELU_F64.to_string()
});
pub(crate) static SHADER_ELU_SIMPLE_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| {
        SHADER_ELU_SIMPLE_F64.to_string()
    });

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

/// Returns the simple ELU WGSL shader (single-pass, no vectorization).
#[must_use]
pub fn wgsl_elu_simple() -> &'static str {
    &SHADER_ELU_SIMPLE_F32
}

/// Default alpha for ELU (matches common framework defaults).
pub const ELU_DEFAULT_ALPHA: f32 = 1.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    alpha: f32,
}

/// ELU: `output = x if x ≥ 0 else α·(eˣ−1)`
pub struct ELU {
    input: Tensor,
    alpha: f32,
}

impl ELU {
    /// Creates ELU with default alpha (1.0).
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self {
            input,
            alpha: ELU_DEFAULT_ALPHA,
        }
    }

    /// Creates ELU with custom alpha.
    #[must_use]
    pub fn with_alpha(input: Tensor, alpha: f32) -> Self {
        Self { input, alpha }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Executes ELU activation and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
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
                label: Some("ELU Params"),
                contents: bytemuck::bytes_of(&Params {
                    size: size as u32,
                    alpha: self.alpha,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout_sig = BindGroupLayoutSignature::reduction();
        let adapter_info = device.adapter_info();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("ELU BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ELU BG"),
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
            Some("ELU Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ELU Pass"),
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
    /// Compute ELU with default alpha (1.0).
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn elu_wgsl(self) -> Result<Self> {
        ELU::new(self).execute()
    }

    /// Compute ELU with a custom alpha.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn elu_wgsl_with_alpha(self, alpha: f32) -> Result<Self> {
        ELU::with_alpha(self, alpha).execute()
    }
}
