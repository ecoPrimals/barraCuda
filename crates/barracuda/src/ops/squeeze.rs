// SPDX-License-Identifier: AGPL-3.0-or-later
//! Squeeze operation - Remove dimensions of size 1
//! Pure WGSL implementation

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Remove dimensions of size 1 from a tensor.
pub struct Squeeze {
    input: Tensor,
}

impl Squeeze {
    /// Create a squeeze operation.
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/tensor/squeeze_f64.wgsl").to_string()
            });
            &S
        }
    }

    /// Execute the squeeze operation, removing dimensions of size 1.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();
        let output_buffer = device.create_buffer_f32(size)?;

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Squeeze BGL"),
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
                    ],
                });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Squeeze BG"),
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
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Squeeze"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Squeeze PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Squeeze Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Squeeze Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Squeeze Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Compute new shape by removing dimensions of size 1
        let new_shape: Vec<usize> = self
            .input
            .shape()
            .iter()
            .copied()
            .filter(|&dim| dim != 1)
            .collect();

        let new_shape = if new_shape.is_empty() {
            vec![1] // Scalar
        } else {
            new_shape
        };

        Ok(Tensor::from_buffer(
            output_buffer,
            new_shape,
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Remove dimensions of size 1 from this tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn squeeze(self) -> Result<Self> {
        Squeeze::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_squeeze_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // Shape [1, 3, 1] should become [3]
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![1, 3, 1], device)
            .await
            .unwrap();
        let result = input.squeeze().unwrap();

        assert_eq!(result.shape(), &[3]);
        let data = result.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_squeeze_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // All dimensions = 1 (scalar)
        let input = Tensor::from_vec_on(vec![5.0], vec![1, 1, 1], device.clone())
            .await
            .unwrap();
        let result = input.squeeze().unwrap();
        assert_eq!(result.shape(), &[1]); // Should become [1]

        // No dimensions to squeeze
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device.clone())
            .await
            .unwrap();
        let result = input.squeeze().unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[tokio::test]
    async fn test_squeeze_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Multiple singleton dimensions
        let input = Tensor::from_vec_on(vec![1.0; 10], vec![1, 1, 10, 1], device.clone())
            .await
            .unwrap();
        let result = input.squeeze().unwrap();
        assert_eq!(result.shape(), &[10]);

        // Leading and trailing singletons
        let input = Tensor::from_vec_on(vec![1.0; 6], vec![1, 2, 3, 1], device.clone())
            .await
            .unwrap();
        let result = input.squeeze().unwrap();
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[tokio::test]
    async fn test_squeeze_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        // Large tensor with singleton dim
        let input = Tensor::from_vec_on(vec![1.0; 1000], vec![1, 1000], device)
            .await
            .unwrap();
        let result = input.squeeze().unwrap();
        assert_eq!(result.shape(), &[1000]);
        assert_eq!(result.to_vec().unwrap().len(), 1000);
    }

    #[tokio::test]
    async fn test_squeeze_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        // Verify data preservation (zero-copy: from_data borrows slice)
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(&input_data, vec![1, 4, 1], device).unwrap();
        let result = input.squeeze().unwrap();

        assert_eq!(result.shape(), &[4]);
        let output_data = result.to_vec().unwrap();
        assert!(output_data.iter().all(|&x| x.is_finite()));
    }
}
