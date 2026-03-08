// SPDX-License-Identifier: AGPL-3.0-or-later
//! LAMB - Layer-wise Adaptive Moments optimizer for Batch training (Pure WGSL)
//!
//! Enables large batch training (e.g., BERT with 64K batch size)
//! Combines Adam with layer-wise adaptation (trust ratio)
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Layer-wise Adaptive Moments optimizer
pub struct Lamb {
    parameters: Tensor,
    gradients: Tensor,
    momentum: Option<Tensor>,
    variance: Option<Tensor>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    step: usize,
    weight_decay: f32,
}

impl Lamb {
    /// Create a LAMB optimizer step with the given parameters and optional momentum/variance.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        parameters: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        step: usize,
        weight_decay: f32,
        momentum: Option<Tensor>,
        variance: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if parameters.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                parameters.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        // Validate learning rate is positive
        if learning_rate <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "lamb",
                "learning_rate must be positive",
            ));
        }

        // Validate betas in valid range
        if !(0.0..1.0).contains(&beta1) {
            return Err(BarracudaError::invalid_op(
                "lamb",
                "beta1 must be in range [0.0, 1.0)",
            ));
        }

        if !(0.0..1.0).contains(&beta2) {
            return Err(BarracudaError::invalid_op(
                "lamb",
                "beta2 must be in range [0.0, 1.0)",
            ));
        }

        // Validate step is positive
        if step == 0 {
            return Err(BarracudaError::invalid_op(
                "lamb",
                "step must be >= 1 (starts at 1, not 0)",
            ));
        }

        // Validate momentum and variance shapes if provided
        if let Some(ref m_tensor) = momentum {
            if m_tensor.shape() != parameters.shape() {
                return Err(BarracudaError::shape_mismatch(
                    m_tensor.shape().to_vec(),
                    parameters.shape().to_vec(),
                ));
            }
        }

        if let Some(ref v_tensor) = variance {
            if v_tensor.shape() != parameters.shape() {
                return Err(BarracudaError::shape_mismatch(
                    v_tensor.shape().to_vec(),
                    parameters.shape().to_vec(),
                ));
            }
        }

        Ok(Self {
            parameters,
            gradients,
            momentum,
            variance,
            learning_rate,
            beta1,
            beta2,
            step,
            weight_decay,
        })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/optimizer/lamb_f64.wgsl"
            ).to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Execute LAMB optimizer step.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.parameters.device();
        let size = self.parameters.shape().iter().product::<usize>();
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;

        // Create writable buffers using GPU copy operations (zero CPU fallbacks)
        let parameters_buffer = device.create_buffer_f32(size)?;

        // Copy parameters buffer using GPU copy
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("LAMB Buffer Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            self.parameters.buffer(),
            0,
            &parameters_buffer,
            0,
            byte_size,
        );

        // Create momentum buffer (GPU copy or zero initialization)
        let m_buffer = if let Some(ref m_tensor) = self.momentum {
            let m_buf = device.create_buffer_f32(size)?;
            encoder.copy_buffer_to_buffer(m_tensor.buffer(), 0, &m_buf, 0, byte_size);
            m_buf
        } else {
            device.create_buffer_f32(size)?
        };

        // Create variance buffer (GPU copy or zero initialization)
        let v_buffer = if let Some(ref v_tensor) = self.variance {
            let v_buf = device.create_buffer_f32(size)?;
            encoder.copy_buffer_to_buffer(v_tensor.buffer(), 0, &v_buf, 0, byte_size);
            v_buf
        } else {
            device.create_buffer_f32(size)?
        };

        // Submit buffer copies
        device.submit_and_poll(Some(encoder.finish()));

        // Create intermediate adam_step buffer
        let adam_step_buffer = device.create_buffer_f32(size)?;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            step: u32,
            learning_rate: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            weight_decay: f32,
            _pad1: u32,
        }

        let params = Params {
            size: size as u32,
            step: self.step as u32,
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: 1e-8,
            weight_decay: self.weight_decay,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LAMB Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("LAMB Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("LAMB Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LAMB Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.gradients.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: m_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adam_step_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LAMB Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        // Create encoder
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("LAMB Encoder"),
        });

        // Step 1: Compute Adam step
        let adam_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("LAMB Adam Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("compute_adam_step"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LAMB Adam Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&adam_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Step 2: Apply trust ratio
        let trust_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("LAMB Trust Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("apply_trust_ratio"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LAMB Trust Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&trust_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            // Trust ratio computation is a reduction over parameters
            let caps = DeviceCapabilities::from_device(device);
            let param_size = self.parameters.shape().iter().product::<usize>();
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (param_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let updated_params = Tensor::from_buffer(
            output_buffer,
            self.parameters.shape().to_vec(),
            device.clone(),
        );

        let updated_m =
            Tensor::from_buffer(m_buffer, self.parameters.shape().to_vec(), device.clone());
        let updated_v =
            Tensor::from_buffer(v_buffer, self.parameters.shape().to_vec(), device.clone());

        Ok((updated_params, updated_m, updated_v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_lamb_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let params = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
            .await
            .unwrap();

        let lamb = Lamb::new(params, gradients, 0.001, 0.9, 0.999, 1, 0.0, None, None).unwrap();
        let (updated_params, _m, _v) = lamb.execute().unwrap();

        assert_eq!(updated_params.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_lamb_with_state() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let params = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
            .await
            .unwrap();

        // Step 1
        let lamb1 = Lamb::new(
            params.clone(),
            gradients.clone(),
            0.001,
            0.9,
            0.999,
            1,
            0.0,
            None,
            None,
        )
        .unwrap();
        let (params1, m1, v1) = lamb1.execute().unwrap();

        // Step 2 with accumulated state
        let lamb2 = Lamb::new(
            params1,
            gradients,
            0.001,
            0.9,
            0.999,
            2,
            0.0,
            Some(m1),
            Some(v1),
        )
        .unwrap();
        let (params2, _m2, _v2) = lamb2.execute().unwrap();

        assert_eq!(params2.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_lamb_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 128;
        let params = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.01; size], vec![size], device.clone())
            .await
            .unwrap();

        let lamb = Lamb::new(params, gradients, 0.001, 0.9, 0.999, 1, 0.0, None, None).unwrap();
        let (updated_params, updated_m, updated_v) = lamb.execute().unwrap();

        assert_eq!(updated_params.shape(), &[size]);
        assert_eq!(updated_m.shape(), &[size]);
        assert_eq!(updated_v.shape(), &[size]);
    }
}
