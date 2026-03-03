// SPDX-License-Identifier: AGPL-3.0-or-later
//! MAE Loss - GPU-accelerated Mean Absolute Error Loss
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (existing shader evolved)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! ## Algorithm
//!
//! ```text
//! MAE = (1/n) * Σ |y_pred - y_true|
//! ```
//!
//! **Key Properties**:
//! - Less sensitive to outliers than MSE
//! - Linear penalty for errors
//! - Robust loss function
//! - Used in regression tasks
//!
//! **Used By**: Robust regression, forecasting, time series
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![1000]).await?;
//! let targets = Tensor::randn(vec![1000]).await?;
//!
//! let loss = predictions.mae_loss(&targets)?;
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MAELossParams {
    reduction_mode: u32,
    size: u32,
    _padding: [u32; 2],
}

pub struct MAELoss {
    predictions: Tensor,
    targets: Tensor,
}

impl MAELoss {
    pub fn new(predictions: Tensor, targets: Tensor) -> Result<Self> {
        // Validate shapes match
        if predictions.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                predictions.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        Ok(Self {
            predictions,
            targets,
        })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/loss/mae_loss.wgsl")
    }

    /// f64 MAE loss (tree reduction for accumulation accuracy).
    pub fn wgsl_shader_f64() -> &'static str {
        include_str!("../shaders/loss/mae_loss_f64.wgsl")
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.shape().iter().product::<usize>();

        let params = MAELossParams {
            reduction_mode: 0, // mean
            size: size as u32,
            _padding: [0; 2],
        };

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mae_loss_output"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mae_loss_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mae_loss_shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mae_loss_bind_group_layout"),
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("mae_loss_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mae_loss_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mae_loss_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.predictions.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.targets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mae_loss_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mae_loss_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.predictions.shape().to_vec(),
            device.clone(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// MAE Loss (Mean Absolute Error) - robust regression loss
    ///
    /// **Deep Debt**: Essential for robust regression tasks
    ///
    /// # Arguments
    /// - `targets`: Target tensor [same shape as predictions]
    ///
    /// # Returns
    /// - Loss tensor [same shape as input]
    ///
    /// # Example
    /// ```rust,ignore
    /// // Regression
    /// let loss = predictions.mae_loss(&targets)?;
    /// ```
    ///
    /// # Note
    /// - Less sensitive to outliers than MSE
    /// - Linear penalty for errors
    /// - Used in robust regression
    pub fn mae_loss(self, targets: &Self) -> Result<Self> {
        MAELoss::new(self, targets.clone())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_mae_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![1.5, 2.5, 3.5, 4.5], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.mae_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), 4);
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x >= 0.0)); // MAE is always non-negative
    }

    #[tokio::test]
    async fn test_mae_loss_perfect() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Perfect predictions should have zero loss
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let loss = predictions.mae_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert!(data.iter().all(|&x| x.abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_mae_loss_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Shape mismatch
        let predictions = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0; 5], vec![5], device.clone())
            .await
            .unwrap();

        assert!(predictions.mae_loss(&targets).is_err());
    }

    #[tokio::test]
    async fn test_mae_loss_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 1000;
        let predictions = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![2.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let loss = predictions.mae_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x.is_finite()));
        // Should be close to 1.0 (absolute difference)
        assert!((data[0] - 1.0).abs() < 0.1);
    }
}
