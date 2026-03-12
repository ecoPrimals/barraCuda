// SPDX-License-Identifier: AGPL-3.0-only
//! `RReLU` — Randomized Leaky `ReLU`, GPU-resident, pipeline-cached, batchable
//!
//! f64 canonical — f32 derived via `downcast_f64_to_f32` when needed.
//!
//! `rrelu(x) = x if x ≥ 0 else a·x` where `a ~ Uniform(lower, upper)` (per-element, seeded)
//!
//! Deep Debt Principles:
//! - Eliminated CPU readback — output stays GPU-resident via buffer pool
//! - Pipeline cached: `GLOBAL_CACHE` eliminates recompilation overhead
//! - Batchable: routes through `TensorContext::record_operation()`

/// f64 is the canonical source.
const SHADER_F64: &str = include_str!("../shaders/activation/rrelu_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

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
    lower: f32,
    upper: f32,
    seed: u32,
}

/// Randomized Leaky `ReLU`.
pub struct RReLU {
    input: Tensor,
    lower: f32,
    upper: f32,
    seed: u32,
}

impl RReLU {
    /// Create a Randomized Leaky `ReLU` operation with the given slope bounds and seed.
    #[must_use]
    pub fn new(input: Tensor, lower: f32, upper: f32, seed: u32) -> Self {
        Self {
            input,
            lower,
            upper,
            seed,
        }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the `RReLU` activation and return the result tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();
        let ctx = get_device_context(device);
        let caps = DeviceCapabilities::from_device(device);
        let wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let workgroups = (size as u32).div_ceil(wg_size);
        let adapter_info = device.adapter_info();

        let output_buffer = ctx.acquire_pooled_output(size);

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RReLU Params"),
                contents: bytemuck::bytes_of(&Params {
                    size: size as u32,
                    lower: self.lower,
                    upper: self.upper,
                    seed: self.seed,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("RReLU BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RReLU BG"),
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
            Some("RReLU Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RReLU Pass"),
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
    /// Apply Randomized Leaky `ReLU` (GPU-resident, pipeline-cached, batchable).
    /// `lower`/`upper` bound the uniform slope for negative values.
    /// Same `seed` always produces the same result — useful for evaluation mode.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn rrelu_wgsl(self, lower: f32, upper: f32, seed: u32) -> Result<Self> {
        RReLU::new(self, lower, upper, seed).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_rrelu_positive_unchanged() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], device);
        let result = input
            .rrelu_wgsl(0.125, 0.333, 42)
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(result, [1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_rrelu_negative_scaled() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(vec![-1.0f32, -2.0], vec![2], device);
        let result = input
            .rrelu_wgsl(0.125, 0.333, 42)
            .unwrap()
            .to_vec()
            .unwrap();
        // Each negative value multiplied by a ∈ [0.125, 0.333]
        assert!(
            result[0] > -0.333 && result[0] < -0.125,
            "r[0]={}",
            result[0]
        );
        assert!(
            result[1] > -0.666 && result[1] < -0.250,
            "r[1]={}",
            result[1]
        );
    }

    #[tokio::test]
    async fn test_rrelu_deterministic_seed() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let make = || {
            Tensor::new(vec![-1.0f32], vec![1], device.clone())
                .rrelu_wgsl(0.125, 0.333, 42)
                .unwrap()
                .to_vec()
                .unwrap()
        };
        assert_eq!(make(), make(), "same seed must produce identical output");
    }
}
