// SPDX-License-Identifier: AGPL-3.0-or-later
//! AdaBound Optimizer - GPU-accelerated Adaptive Gradient Methods with Dynamic Bound
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Adaptive learning rate optimizer with dynamic bound on learning rates
//! Smoothly transitions from adaptive methods to SGD
//!
//! Reference: "Adaptive Gradient Methods with Dynamic Bound of Learning Rate" by Luo et al. (2019)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AdaBoundParams {
    size: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    final_lr: f32,
    gamma: f32,
    step: u32,
}

/// AdaBound optimizer — adaptive gradients with dynamic learning rate bounds.
pub struct AdaBound {
    gradients: Tensor,
    params: Tensor,
    m: Option<Tensor>,
    v: Option<Tensor>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    final_lr: f32,
    gamma: f32,
    step: usize,
}

impl AdaBound {
    /// Create AdaBound optimizer. Smoothly transitions from Adam to SGD.
    pub fn new(
        params: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        final_lr: f32,
        gamma: f32,
        step: usize,
        m: Option<Tensor>,
        v: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if params.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                params.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        // Validate learning rate is positive
        if learning_rate <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "adabound",
                "learning_rate must be positive",
            ));
        }

        // Validate betas in valid range
        if !(0.0..1.0).contains(&beta1) {
            return Err(BarracudaError::invalid_op(
                "adabound",
                "beta1 must be in range [0.0, 1.0)",
            ));
        }

        if !(0.0..1.0).contains(&beta2) {
            return Err(BarracudaError::invalid_op(
                "adabound",
                "beta2 must be in range [0.0, 1.0)",
            ));
        }

        // Validate step is positive
        if step == 0 {
            return Err(BarracudaError::invalid_op(
                "adabound",
                "step must be >= 1 (starts at 1, not 0)",
            ));
        }

        // Validate m and v shapes if provided
        if let Some(ref m_tensor) = m {
            if m_tensor.shape() != params.shape() {
                return Err(BarracudaError::shape_mismatch(
                    m_tensor.shape().to_vec(),
                    params.shape().to_vec(),
                ));
            }
        }

        if let Some(ref v_tensor) = v {
            if v_tensor.shape() != params.shape() {
                return Err(BarracudaError::shape_mismatch(
                    v_tensor.shape().to_vec(),
                    params.shape().to_vec(),
                ));
            }
        }

        Ok(Self {
            gradients,
            params,
            m,
            v,
            learning_rate,
            beta1,
            beta2,
            final_lr,
            gamma,
            step,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/optimizer/adabound_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute AdaBound step. Returns (updated_params, m, v).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.params.device();
        let size = self.params.shape().iter().product::<usize>();

        let adabound_params = AdaBoundParams {
            size: size as u32,
            lr: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            final_lr: self.final_lr,
            gamma: self.gamma,
            step: self.step as u32,
        };

        // Create writable buffers using GPU copy operations (zero CPU fallbacks)
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;

        // Copy params to writable buffer using GPU copy
        let params_buffer = device.create_buffer_f32(size)?;
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("AdaBound Buffer Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(self.params.buffer(), 0, &params_buffer, 0, byte_size);

        // Copy or create m buffer (GPU copy or zero initialization)
        let m_buffer = if let Some(ref m_tensor) = self.m {
            let m_buf = device.create_buffer_f32(size)?;
            encoder.copy_buffer_to_buffer(m_tensor.buffer(), 0, &m_buf, 0, byte_size);
            m_buf
        } else {
            device.create_buffer_f32(size)?
        };

        // Copy or create v buffer (GPU copy or zero initialization)
        let v_buffer = if let Some(ref v_tensor) = self.v {
            let v_buf = device.create_buffer_f32(size)?;
            encoder.copy_buffer_to_buffer(v_tensor.buffer(), 0, &v_buf, 0, byte_size);
            v_buf
        } else {
            device.create_buffer_f32(size)?
        };

        // Submit buffer copies
        device.submit_and_poll(Some(encoder.finish()));

        let adabound_params_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adabound_params"),
                    contents: bytemuck::cast_slice(&[adabound_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("adabound_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("adabound_bind_group_layout"),
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

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("adabound_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adabound_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adabound_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.gradients.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adabound_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("adabound_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adabound_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let updated_params =
            Tensor::from_buffer(params_buffer, self.params.shape().to_vec(), device.clone());

        let updated_m = Tensor::from_buffer(m_buffer, self.params.shape().to_vec(), device.clone());

        let updated_v = Tensor::from_buffer(v_buffer, self.params.shape().to_vec(), device.clone());

        Ok((updated_params, updated_m, updated_v))
    }
}

impl Tensor {
    /// AdaBound optimizer step
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as params]
    /// - `learning_rate`: Initial learning rate
    /// - `beta1`: Exponential decay for first moment, typically 0.9
    /// - `beta2`: Exponential decay for second moment, typically 0.999
    /// - `final_lr`: Final learning rate for SGD convergence
    /// - `gamma`: Rate of convergence to SGD
    /// - `step`: Current iteration (starts at 1, not 0)
    /// - `m`: First moment estimate (None for first step)
    /// - `v`: Second moment estimate (None for first step)
    ///
    /// # Returns
    /// - Tuple: (updated_params, updated_m, updated_v)
    pub fn adabound_step(
        self,
        gradients: &Self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        final_lr: f32,
        gamma: f32,
        step: usize,
        m: Option<&Self>,
        v: Option<&Self>,
    ) -> Result<(Self, Self, Self)> {
        AdaBound::new(
            self,
            gradients.clone(),
            learning_rate,
            beta1,
            beta2,
            final_lr,
            gamma,
            step,
            m.cloned(),
            v.cloned(),
        )?
        .execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_adabound_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let params = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
            .await
            .unwrap();

        let (updated_params, _m, _v) = params
            .adabound_step(&gradients, 0.001, 0.9, 0.999, 0.01, 0.1, 1, None, None)
            .unwrap();
        let result = updated_params.to_vec().unwrap();

        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
