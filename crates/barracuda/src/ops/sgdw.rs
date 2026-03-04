// SPDX-License-Identifier: AGPL-3.0-or-later
//! SGDW - SGD with Decoupled Weight Decay (Pure WGSL)
//!
//! More principled weight decay than L2 regularization
//! Decouples weight decay from gradient-based update
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

/// f64 is the canonical source.
const SHADER_F64: &str = include_str!("../shaders/optimizer/sgdw_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// SGD with Decoupled Weight Decay
pub struct SGDW {
    parameters: Tensor,
    gradients: Tensor,
    velocity: Option<Tensor>,
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
}

impl SGDW {
    pub fn new(
        parameters: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        dampening: f32,
        nesterov: bool,
        velocity: Option<Tensor>,
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
                "sgdw",
                "learning_rate must be positive",
            ));
        }

        // Validate velocity shape if provided
        if let Some(ref v_tensor) = velocity {
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
            velocity,
            learning_rate,
            momentum,
            weight_decay,
            dampening,
            nesterov,
        })
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.parameters.device();
        let size = self.parameters.shape().iter().product::<usize>();
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;

        // Create writable buffers using GPU copy operations (zero CPU fallbacks)
        let parameters_buffer = device.create_buffer_f32(size)?;

        // Copy parameters buffer using GPU copy
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SGDW Buffer Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            self.parameters.buffer(),
            0,
            &parameters_buffer,
            0,
            byte_size,
        );

        // Create velocity buffer (GPU copy or zero initialization)
        let v_buffer = if let Some(ref v_tensor) = self.velocity {
            let v_buf = device.create_buffer_f32(size)?;
            encoder.copy_buffer_to_buffer(v_tensor.buffer(), 0, &v_buf, 0, byte_size);
            v_buf
        } else {
            device.create_buffer_f32(size)?
        };

        // Submit buffer copies
        device.submit_and_poll(Some(encoder.finish()));

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            learning_rate: f32,
            momentum: f32,
            weight_decay: f32,
            dampening: f32,
            nesterov: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: size as u32,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
            dampening: self.dampening,
            nesterov: if self.nesterov { 1 } else { 0 },
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SGDW Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("SGDW Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SGDW Bind Group Layout"),
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
                    ],
                });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SGDW Bind Group"),
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
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SGDW Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        // Create pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SGDW Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SGDW Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SGDW Pass"),
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

        device.submit_and_poll(Some(encoder.finish()));

        let updated_params = Tensor::from_buffer(
            output_buffer,
            self.parameters.shape().to_vec(),
            device.clone(),
        );

        let updated_v =
            Tensor::from_buffer(v_buffer, self.parameters.shape().to_vec(), device.clone());

        Ok((updated_params, updated_v))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_sgdw_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let params = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
            .await
            .unwrap();

        let sgdw = SGDW::new(params, gradients, 0.01, 0.9, 0.0001, 0.0, false, None).unwrap();
        let (updated_params, _v) = sgdw.execute().unwrap();

        assert_eq!(updated_params.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_sgdw_with_momentum() {
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
        let sgdw1 = SGDW::new(
            params.clone(),
            gradients.clone(),
            0.01,
            0.9,
            0.0001,
            0.0,
            false,
            None,
        )
        .unwrap();
        let (params1, v1) = sgdw1.execute().unwrap();

        // Step 2 with velocity
        let sgdw2 = SGDW::new(params1, gradients, 0.01, 0.9, 0.0001, 0.0, false, Some(v1)).unwrap();
        let (params2, _v2) = sgdw2.execute().unwrap();

        assert_eq!(params2.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_sgdw_nesterov() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let params = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
            .await
            .unwrap();

        let sgdw = SGDW::new(params, gradients, 0.01, 0.9, 0.0001, 0.0, true, None).unwrap();
        let (updated_params, _v) = sgdw.execute().unwrap();

        assert_eq!(updated_params.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_sgdw_large_batch() {
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

        let sgdw = SGDW::new(params, gradients, 0.01, 0.9, 0.0001, 0.0, false, None).unwrap();
        let (updated_params, updated_v) = sgdw.execute().unwrap();

        assert_eq!(updated_params.shape(), &[size]);
        assert_eq!(updated_v.shape(), &[size]);
    }
}
