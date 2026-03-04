// SPDX-License-Identifier: AGPL-3.0-or-later
//! Element-wise subtraction
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (universal compute)
//! - ✅ Capability-based dispatch (vendor-optimized)
//!
//! Formula: C = A - B (element-wise)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const WGSL_SUB_F64: &str = include_str!("../shaders/math/elementwise_sub_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(WGSL_SUB_F64));

pub struct Sub {
    lhs: Tensor,
    rhs: Tensor,
}

impl Sub {
    pub fn new(lhs: Tensor, rhs: Tensor) -> Result<Self> {
        if lhs.shape() != rhs.shape() {
            return Err(BarracudaError::shape_mismatch(
                lhs.shape().to_vec(),
                rhs.shape().to_vec(),
            ));
        }
        Ok(Self { lhs, rhs })
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size = self.lhs.len();
        let output_buffer = device.create_buffer_f32(size)?;

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Sub BGL"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sub BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.lhs.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.rhs.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Sub"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sub PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sub Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Sub Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sub Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.lhs.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn sub(&self, other: &Tensor) -> Result<Self> {
        Sub::new(self.clone(), other.clone())?.execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_sub_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let lhs = Tensor::from_vec_on(vec![10.0, 20.0, 30.0], vec![3], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        assert!((result[0] - 9.0).abs() < 1e-5);
        assert!((result[1] - 18.0).abs() < 1e-5);
        assert!((result[2] - 27.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_sub_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let lhs = Tensor::from_vec_on(vec![0.0, 1e-6, -1e-6, 1.0, -1.0], vec![5], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![0.0, 1e-6, -1e-6, 1.0, -1.0], vec![5], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        assert!(result[0].abs() < 1e-12);
        assert!(result[1].abs() < 1e-12);
        assert!(result[2].abs() < 1e-12);
    }

    #[tokio::test]
    async fn test_sub_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let lhs = Tensor::from_vec_on(vec![f32::INFINITY, 1e10, 0.0], vec![3], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![100.0, 100.0, 100.0], vec![3], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        assert!(result[0].is_infinite() && result[0].is_sign_positive());
        assert_eq!(result[2], -100.0);
    }

    #[tokio::test]
    async fn test_sub_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 1000;
        let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let lhs = Tensor::from_vec_on(lhs_data, vec![size], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(rhs_data, vec![size], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        for (i, &val) in result.iter().enumerate() {
            assert!((val - i as f32).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_sub_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let lhs_data = vec![5.0, 2.5, 1.0, 0.0, -1.0];
        let rhs_data = vec![2.0, 1.5, 0.5, 0.0, -0.5];

        let lhs = Tensor::from_vec_on(lhs_data.clone(), vec![5], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(rhs_data.clone(), vec![5], device)
            .await
            .unwrap();

        let gpu_result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        let cpu_result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-6,
                "Error at {}: GPU={}, CPU={}",
                i,
                gpu,
                cpu
            );
        }
    }
}
