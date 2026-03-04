// SPDX-License-Identifier: AGPL-3.0-or-later
//! LeakyReLU — GPU-resident, pipeline-cached, batchable
//!
//! f64 canonical — f32 derived via downcast_f64_to_f32 when needed.
//!
//! Deep Debt Principles:
//! - Zero hardcoding: Capability-based workgroup dispatch
//! - Batchable: routes through TensorContext::record_operation()
//! - Zero-copy output: buffer pool, no GPU→CPU→GPU round-trip
//! - Pipeline cached: GLOBAL_CACHE eliminates recompilation overhead
//! - Params fixed (S14): Rust `Params` matches WGSL `{ size, negative_slope }`

/// f64 is the canonical source.
const SHADER_F64: &str = include_str!("../shaders/activation/leaky_relu_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

/// f64 canonical — f32 derived via downcast when needed.
const WGSL_LEAKY_RELU_SIMPLE_F64: &str =
    include_str!("../shaders/activation/leaky_relu_simple_f64.wgsl");

/// Simple LeakyReLU variant (single-pass, no vectorization).
pub static WGSL_LEAKY_RELU_SIMPLE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32(WGSL_LEAKY_RELU_SIMPLE_F64)
});

/// Default negative slope for LeakyReLU (matches common framework defaults).
pub const LEAKY_RELU_DEFAULT_SLOPE: f32 = 0.01;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    negative_slope: f32,
}

/// LeakyReLU: `output = x if x ≥ 0 else α·x`
pub struct LeakyRelu {
    input: Tensor,
    negative_slope: f32,
}

impl LeakyRelu {
    pub fn new(input: Tensor) -> Self {
        Self {
            input,
            negative_slope: LEAKY_RELU_DEFAULT_SLOPE,
        }
    }

    pub fn with_slope(input: Tensor, negative_slope: f32) -> Self {
        Self {
            input,
            negative_slope,
        }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

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
                label: Some("LeakyReLU Params"),
                contents: bytemuck::bytes_of(&Params {
                    size: size as u32,
                    negative_slope: self.negative_slope,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout_sig = BindGroupLayoutSignature::reduction();
        let adapter_info = device.adapter_info();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("LeakyReLU BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LeakyReLU BG"),
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
            Some("LeakyReLU Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LeakyReLU Pass"),
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
    /// Compute LeakyReLU with default slope (0.01).
    pub fn leaky_relu_wgsl(self) -> Result<Self> {
        LeakyRelu::new(self).execute()
    }

    /// Compute LeakyReLU with a custom negative slope α.
    pub fn leaky_relu_wgsl_with_slope(self, negative_slope: f32) -> Result<Self> {
        LeakyRelu::with_slope(self, negative_slope).execute()
    }
}
