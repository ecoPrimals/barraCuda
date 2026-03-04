// SPDX-License-Identifier: AGPL-3.0-or-later
//! Boltzmann Sampling (f64) — wateringHole V69
//!
//! GPU-accelerated Boltzmann (softmax) sampling with temperature.

use std::sync::Arc;

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};

/// WGSL shader for Boltzmann sampling.
pub const WGSL_BOLTZMANN_SAMPLING_F64: &str =
    include_str!("../shaders/special/boltzmann_sampling_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    batch_size: u32,
    n_classes: u32,
}

/// GPU-backed Boltzmann sampling.
pub struct BoltzmannSamplingGpu;

impl BoltzmannSamplingGpu {
    /// Execute Boltzmann sampling on GPU.
    ///
    /// Uses Gumbel-max trick: sample = argmax(logits/temp + Gumbel).
    ///
    /// # Arguments
    /// * `logits` - Flattened [batch_size, n_classes]
    /// * `batch_size` - Number of batch elements
    /// * `temperature` - Softmax temperature
    /// * `seed` - Random seed
    pub fn execute(
        device: Arc<WgpuDevice>,
        logits: &[f64],
        batch_size: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let n = logits.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "logits cannot be empty".to_string(),
            });
        }
        if batch_size == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "batch_size must be > 0".to_string(),
            });
        }
        let n_classes = n / batch_size;
        if n_classes * batch_size != n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "logits.len() must be divisible by batch_size: {} / {}",
                    n, batch_size
                ),
            });
        }

        let temp_safe = if temperature > 1e-10 {
            temperature
        } else {
            1.0
        };

        let mut seeds: Vec<u32> = (0..batch_size * 4)
            .map(|i| {
                let s = seed.wrapping_add(i as u64);
                (s ^ (s >> 32)) as u32
            })
            .collect();
        if seeds.len() < 4 {
            seeds.resize(4, 0);
        }

        let logits_buf = device.create_buffer_f64_init("boltzmann:logits", logits);
        let seeds_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("boltzmann:seeds"),
                contents: bytemuck::cast_slice(&seeds),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let out_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boltzmann:output"),
            size: (batch_size * std::mem::size_of::<u32>()) as u64, // batch_size elements
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            batch_size: batch_size as u32,
            n_classes: n_classes as u32,
        };
        let params_buf = device.create_uniform_buffer("boltzmann:params", &params);
        let temp_buf = device.create_buffer_f64_init("boltzmann:temp", &[temp_safe]);

        ComputeDispatch::new(&device, "boltzmann_sampling")
            .shader(WGSL_BOLTZMANN_SAMPLING_F64, "main")
            .f64()
            .storage_read(0, &logits_buf)
            .storage_rw(1, &seeds_buf)
            .storage_rw(2, &out_buf)
            .uniform(3, &params_buf)
            .storage_read(4, &temp_buf)
            .dispatch(
                batch_size.div_ceil(WORKGROUP_SIZE_COMPACT as usize) as u32,
                1,
                1,
            )
            .submit();

        let indices = device.read_buffer_u32(&out_buf, batch_size)?;
        Ok(indices)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    #[ignore = "compile_shader_f64 produces vec2<f32>/vec2<f64> type confusion on some drivers (llvmpipe); run with BARRACUDA_TEST_BACKEND=gpu on f64-capable hardware"]
    async fn test_boltzmann_construction() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.0];

        let indices = BoltzmannSamplingGpu::execute(device, &logits, 1, 1.0, 42).unwrap();
        assert_eq!(indices.len(), 1);
        assert!(indices[0] < 5);
    }

    #[tokio::test]
    async fn test_boltzmann_dimension_check() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let logits = vec![];

        let result = BoltzmannSamplingGpu::execute(device, &logits, 1, 1.0, 42);
        assert!(result.is_err());
    }
}
