// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leave-One-Out Cross-Validation for kernel methods - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// Leave-one-out cross-validation residuals for kernel methods.
/// `LOO_i` = (`y_i` - `pred_i`) / (1 - `H_ii`)
pub struct LooCv {
    hat_matrix: Tensor,
    y: Tensor,
    predictions: Tensor,
}

impl LooCv {
    /// Create LOO-CV residuals from hat matrix, targets, and predictions.
    #[must_use]
    pub fn new(hat_matrix: Tensor, y: Tensor, predictions: Tensor) -> Self {
        Self {
            hat_matrix,
            y,
            predictions,
        }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/interpolation/loo_cv_f64.wgsl"
            ))
        });
        SHADER.as_str()
    }

    /// Compute leave-one-out cross-validation residuals.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.hat_matrix.device();
        let n: usize = self.y.shape().iter().product();

        if n == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        let output_buffer = device.create_buffer_f32(n)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
        }

        let params = Params { n: n as u32 };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LOO-CV Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("LOO-CV Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
            label: Some("LOO-CV Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.hat_matrix.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.y.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.predictions.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("LOO-CV"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LOO-CV Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LOO-CV Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("LOO-CV Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LOO-CV Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(n as u32);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(output_buffer, vec![n], device.clone()))
    }
}

impl Tensor {
    /// Compute LOO-CV residuals: (y - pred) / (1 - diag(H))
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn loo_cv(self, y: Tensor, predictions: Tensor) -> Result<Self> {
        LooCv::new(self, y, predictions).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_loo_cv() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Simple 2x2 case: H = [[0.5, 0.5], [0.5, 0.5]], y = [1, 2], pred = [0.8, 1.7]
        // LOO_0 = (1 - 0.8) / (1 - 0.5) = 0.2/0.5 = 0.4
        // LOO_1 = (2 - 1.7) / (1 - 0.5) = 0.3/0.5 = 0.6
        let hat_matrix = vec![0.5f32, 0.5, 0.5, 0.5];
        let y = vec![1.0f32, 2.0];
        let pred = vec![0.8f32, 1.7];

        let hat = Tensor::new(hat_matrix, vec![2, 2], device.clone());
        let y_t = Tensor::new(y, vec![2], device.clone());
        let pred_t = Tensor::new(pred, vec![2], device.clone());

        let output = hat.loo_cv(y_t, pred_t).unwrap();
        let result = output.to_vec().unwrap();

        assert!((result[0] - 0.4).abs() < 1e-5);
        assert!((result[1] - 0.6).abs() < 1e-5);
    }
}
