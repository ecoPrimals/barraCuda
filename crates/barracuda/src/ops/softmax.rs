// SPDX-License-Identifier: AGPL-3.0-or-later
//! Softmax activation — GPU-resident, pipeline-cached, batchable
//!
//! Formula: `softmax(x_i) = exp(x_i) / Σ exp(x_j)`
//! Logical-size uniform prevents incorrect normalisation over pooled buffers (S14-007).

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// f64 canonical — simple softmax (single workgroup).
const SHADER_SOFTMAX_SIMPLE_F64: &str =
    include_str!("../shaders/activation/softmax_simple_f64.wgsl");
pub(crate) static SHADER_SOFTMAX_SIMPLE_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| {
        crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(
            SHADER_SOFTMAX_SIMPLE_F64,
        )
    });

/// f64 canonical — basic softmax (multi-pass).
const WGSL_SOFTMAX_BASIC_F64: &str = include_str!("../shaders/activation/softmax_f64.wgsl");
pub static WGSL_SOFTMAX_BASIC: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(WGSL_SOFTMAX_BASIC_F64)
});

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
}

/// Softmax activation operation
pub struct Softmax {
    input: Tensor,
}

impl Softmax {
    /// Create Softmax operation
    pub fn new(input: Tensor) -> Result<Self> {
        // Softmax expects 1D or last dimension for now
        if input.shape().is_empty() {
            return Err(BarracudaError::invalid_op(
                "Softmax",
                "Empty tensor not supported",
            ));
        }
        Ok(Self { input })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        &SHADER_SOFTMAX_SIMPLE_F32
    }

    /// Execute Softmax — GPU-resident, pipeline-cached, batchable.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        let ctx = get_device_context(device);
        let caps = DeviceCapabilities::from_device(device);
        let wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
        let workgroups = (size as u32).div_ceil(wg_size).max(1);

        // Pooled output prevents oversized-buffer bug (S14-007 fix stays correct
        // because we pass `size` as the logical uniform, not arrayLength).
        let output_buffer = ctx.acquire_pooled_output(size);

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Softmax Params"),
                contents: bytemuck::bytes_of(&Params { size: size as u32 }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout_sig = BindGroupLayoutSignature::reduction();
        let adapter_info = device.adapter_info();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("Softmax BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Softmax BG"),
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
            Some("Softmax Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Softmax Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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

// Convenience method on Tensor
impl Tensor {
    /// Apply Softmax activation.
    pub fn softmax(self) -> Result<Self> {
        tracing::debug!("Routing softmax to WGSL");
        Softmax::new(self)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_softmax_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();
        let output = input.softmax().unwrap();
        let result = output.to_vec().unwrap();

        // Sum = 1, all in (0,1), monotonic
        assert!(result.iter().all(|&x| x > 0.0 && x < 1.0));
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[2] > result[1] && result[1] > result[0]);
    }

    #[tokio::test]
    async fn test_softmax_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1e-6, 2e-6, 3e-6], vec![3], device)
            .await
            .unwrap();
        let output = input.softmax().unwrap();
        let result = output.to_vec().unwrap();

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_softmax_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![100.0, 200.0, 300.0], vec![3], device)
            .await
            .unwrap();
        let output = input.softmax().unwrap();
        let result = output.to_vec().unwrap();

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[2] > 0.99); // Largest dominates
    }

    #[tokio::test]
    async fn test_softmax_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 1000;
        let input_data: Vec<f32> = (0..size).map(|i| (i as f32) / 10.0).collect();
        let input = Tensor::from_vec_on(input_data, vec![size], device)
            .await
            .unwrap();
        let output = input.softmax().unwrap();
        let result = output.to_vec().unwrap();

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_softmax_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        fn softmax_cpu(x: &[f32]) -> Vec<f32> {
            let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|&e| e / sum).collect()
        }

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let output = input.softmax().unwrap();
        let gpu_result = output.to_vec().unwrap();
        let cpu_result = softmax_cpu(&input_data);

        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-5,
                "Error at {}: GPU={}, CPU={}",
                i,
                gpu,
                cpu
            );
        }
    }
}
