// SPDX-License-Identifier: AGPL-3.0-or-later
//! AdaGrad Optimizer - GPU-accelerated Adaptive Gradient Algorithm
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (existing shader evolved)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)
//!
//! ## Algorithm
//!
//! ```text
//! G_t = G_{t-1} + g_t²
//! w_t = w_{t-1} - (lr / sqrt(G_t + ε)) * g_t
//! ```
//!
//! **Key Properties**:
//! - Adapts learning rate per parameter based on gradient history
//! - Accumulates squared gradients
//! - Good for sparse gradients
//! - Learning rate monotonically decreases (limitation)
//!
//! **Parameters**:
//! - `learning_rate`: Initial step size, typically 0.01
//! - `epsilon` (ε): Numerical stability constant, typically 1e-8
//!
//! **Used By**: Sparse gradient problems, NLP tasks
//!
//! **Note**: RMSprop and Adam address AdaGrad's monotonic decrease issue
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let weights = Tensor::randn(vec![1000]).await?;
//! let gradients = Tensor::randn(vec![1000]).await?;
//!
//! // First step
//! let (w1, acc1) = weights.adagrad_step(&gradients, 0.01, None)?;
//!
//! // Subsequent steps
//! let (w2, acc2) = w1.adagrad_step(&gradients, 0.01, Some(&acc1))?;
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AdaGradParams {
    learning_rate: f32,
    epsilon: f32,
    weight_decay: f32,
    _padding: u32,
}

pub struct AdaGrad {
    weights: Tensor,
    gradients: Tensor,
    accumulated: Option<Tensor>,
    learning_rate: f32,
}

impl AdaGrad {
    pub fn new(
        weights: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        accumulated: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if weights.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                weights.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        // Validate learning rate is positive
        if learning_rate <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "adagrad",
                "learning_rate must be positive",
            ));
        }

        // Validate accumulated shape if provided
        if let Some(ref acc) = accumulated {
            if acc.shape() != weights.shape() {
                return Err(BarracudaError::shape_mismatch(
                    acc.shape().to_vec(),
                    weights.shape().to_vec(),
                ));
            }
        }

        Ok(Self {
            weights,
            gradients,
            accumulated,
            learning_rate,
        })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/optimizer/adagrad_f64.wgsl"
            ))
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.weights.device();
        let size = self.weights.shape().iter().product::<usize>();

        let params = AdaGradParams {
            learning_rate: self.learning_rate,
            epsilon: 1e-8,
            weight_decay: 0.0,
            _padding: 0,
        };

        // Create accumulated buffer if not provided
        let zeros = vec![0.0f32; size];
        let accumulated_in = if let Some(ref acc_tensor) = self.accumulated {
            acc_tensor.buffer()
        } else {
            &device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adagrad_acc_zeros"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let weights_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adagrad_weights_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accumulated_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adagrad_accumulated_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adagrad_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("adagrad_shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("adagrad_bind_group_layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
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
                    label: Some("adagrad_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adagrad_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adagrad_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.weights.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.gradients.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: accumulated_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weights_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: accumulated_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adagrad_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adagrad_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let updated_weights = Tensor::from_buffer(
            weights_out_buffer,
            self.weights.shape().to_vec(),
            device.clone(),
        );

        let updated_accumulated = Tensor::from_buffer(
            accumulated_out_buffer,
            self.weights.shape().to_vec(),
            device.clone(),
        );

        Ok((updated_weights, updated_accumulated))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// AdaGrad optimizer step - adaptive learning rate for sparse gradients
    ///
    /// **Deep Debt**: Historical optimizer, foundation for Adam/RMSprop
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as weights]
    /// - `learning_rate`: Initial step size, typically 0.01
    /// - `accumulated`: Accumulated squared gradients (None for first step)
    ///
    /// # Returns
    /// - Tuple: (updated_weights, updated_accumulated)
    ///
    /// # Example
    /// ```rust,ignore
    /// // First step
    /// let (w1, acc1) = weights.adagrad_step(&grads, 0.01, None)?;
    ///
    /// // Subsequent steps
    /// let (w2, acc2) = w1.adagrad_step(&grads, 0.01, Some(&acc1))?;
    /// ```
    ///
    /// # Note
    /// - Good for sparse gradients
    /// - Learning rate monotonically decreases
    /// - RMSprop and Adam address this limitation
    /// - learning_rate must be positive
    pub fn adagrad_step(
        self,
        gradients: &Self,
        learning_rate: f32,
        accumulated: Option<&Self>,
    ) -> Result<(Self, Self)> {
        AdaGrad::new(self, gradients.clone(), learning_rate, accumulated.cloned())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_adagrad_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
            .await
            .unwrap();

        let (updated_weights, updated_acc) = weights.adagrad_step(&gradients, 0.01, None).unwrap();

        let result = updated_weights.to_vec().unwrap();
        let acc = updated_acc.to_vec().unwrap();

        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&x| x.is_finite()));
        assert!(acc.iter().all(|&x| x >= 0.0));
        assert!(result[0] < 1.0, "Expected descent, got {}", result[0]);
    }

    #[tokio::test]
    async fn test_adagrad_accumulation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
            .await
            .unwrap();

        // Step 1
        let (weights1, acc1) = weights.adagrad_step(&gradients, 0.01, None).unwrap();
        let acc_data1 = acc1.to_vec().unwrap();
        assert!(acc_data1.iter().all(|&x| x > 0.0));

        // Step 2 with accumulated state
        let (weights2, acc2) = weights1
            .adagrad_step(&gradients, 0.01, Some(&acc1))
            .unwrap();
        let result = weights2.to_vec().unwrap();
        let acc_data2 = acc2.to_vec().unwrap();

        assert!(result.iter().all(|&x| x.is_finite()));
        // Accumulated should increase
        assert!(acc_data2.iter().zip(&acc_data1).all(|(&a2, &a1)| a2 > a1));
    }

    #[tokio::test]
    async fn test_adagrad_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();
        let gradients = Tensor::from_vec_on(vec![0.1; 5], vec![5], device.clone())
            .await
            .unwrap();
        let grads_correct = Tensor::from_vec_on(vec![0.1; 10], vec![10], device.clone())
            .await
            .unwrap();

        // Shape mismatch
        assert!(weights
            .clone()
            .adagrad_step(&gradients, 0.01, None)
            .is_err());

        // Invalid learning rate
        assert!(weights
            .clone()
            .adagrad_step(&grads_correct, -0.01, None)
            .is_err());
        assert!(weights.adagrad_step(&grads_correct, 0.0, None).is_err());
    }

    #[tokio::test]
    async fn test_adagrad_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 128;
        let weights = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.01; size], vec![size], device.clone())
            .await
            .unwrap();

        let (updated_weights, updated_acc) = weights.adagrad_step(&gradients, 0.01, None).unwrap();

        let result = updated_weights.to_vec().unwrap();
        let acc = updated_acc.to_vec().unwrap();

        assert_eq!(result.len(), size);
        assert_eq!(acc.len(), size);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_adagrad_multi_step() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![10.0, 20.0], vec![2], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
            .await
            .unwrap();

        // Step 1
        let (weights1, acc1) = weights.adagrad_step(&gradients, 0.1, None).unwrap();
        let result1 = weights1.to_vec().unwrap();

        assert!(result1[0] < 10.0, "Expected descent, got {}", result1[0]);
        assert!(result1[1] < 20.0, "Expected descent, got {}", result1[1]);

        // Step 2 with accumulated state
        let (weights2, _acc2) = weights1.adagrad_step(&gradients, 0.1, Some(&acc1)).unwrap();
        let result2 = weights2.to_vec().unwrap();

        // Should continue descending
        assert!(result2[0] < result1[0]);
        assert!(result2[1] < result1[1]);
    }
}
