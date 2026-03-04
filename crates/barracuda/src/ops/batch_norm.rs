// SPDX-License-Identifier: AGPL-3.0-or-later
//! BatchNorm operation - Batch normalization
//! Pure WGSL implementation
//!
//! Deep Debt Principles:
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 canonical — BatchNorm with running mean/var (training mode, CNN).
const WGSL_BATCHNORM_TRAINING_F64: &str = include_str!("../shaders/norm/batchnorm_f64.wgsl");
pub static WGSL_BATCHNORM_TRAINING: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(WGSL_BATCHNORM_TRAINING_F64)
});

/// f64 canonical — 2D batch normalization (NCHW format, per-channel stats).
const WGSL_BATCH_NORM_2D_F64: &str = include_str!("../shaders/norm/batch_norm2d_f64.wgsl");
pub static WGSL_BATCH_NORM_2D: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(WGSL_BATCH_NORM_2D_F64)
});

/// f64 canonical — per-tensor batch norm (simplified).
const SHADER_BATCH_NORM_F64: &str = include_str!("../shaders/norm/batch_norm_f64.wgsl");
static SHADER_BATCH_NORM_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_BATCH_NORM_F64)
});

/// GPU shader for group normalization (groups within channels).
pub fn wgsl_groupnorm() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
            "../shaders/norm/groupnorm_f64.wgsl"
        ))
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// GPU shader for instance normalization (per-instance per-channel).
pub fn wgsl_instancenorm() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
            "../shaders/norm/instancenorm_f64.wgsl"
        ))
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchNormParams {
    epsilon: f32,
    _padding: [f32; 7],
}

pub struct BatchNorm {
    input: Tensor,
    epsilon: f32,
}

impl BatchNorm {
    pub fn new(input: Tensor, epsilon: f32) -> Self {
        Self { input, epsilon }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_BATCH_NORM_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();
        let output_buffer = device.create_buffer_f32(size)?;

        let params = BatchNormParams {
            epsilon: self.epsilon,
            _padding: [0.0; 7],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchNorm Params"),
            size: std::mem::size_of::<BatchNormParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("BatchNorm BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BatchNorm BG"),
            layout: &bind_group_layout,
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
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("BatchNorm"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("BatchNorm PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BatchNorm Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("BatchNorm Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BatchNorm Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch (vendor-optimized workgroups)
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn batch_norm(self, epsilon: f32) -> Result<Self> {
        BatchNorm::new(self, epsilon).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn batch_norm_cpu(input: &[f32], epsilon: f32) -> Vec<f32> {
        let n = input.len() as f32;
        let mean: f32 = input.iter().sum::<f32>() / n;
        let variance: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = (variance + epsilon).sqrt();
        input.iter().map(|x| (x - mean) / std).collect()
    }

    #[tokio::test]
    async fn test_batch_norm_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();

        let data = result.to_vec().unwrap();
        assert_eq!(data.len(), 4);

        let expected = batch_norm_cpu(&input_data, 1e-5);
        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // All same values (zero variance)
        let input_data = vec![5.0, 5.0, 5.0, 5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        // Should be all zeros (normalized to mean)
        for val in data.iter() {
            assert!(val.abs() < 1e-3);
        }

        // Negative values
        let input_data = vec![-2.0, -1.0, 1.0, 2.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);
        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Single element
        let input_data = vec![5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![1], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        assert!(data[0].abs() < 1e-3); // Should be ~0

        // Wide range of values
        let input_data = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);
        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 1000 elements
        let input_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![1000], device)
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();

        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);

        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-3);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test FP32 precision
        let input_data = vec![1.234, 5.678, 9.012, 3.456, 7.890];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();

        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);

        // Verify FP32 precision
        let max_error = data
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| (r - e).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 1e-4,
            "Max error: {} exceeds FP32 threshold",
            max_error
        );
    }
}
