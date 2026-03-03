// SPDX-License-Identifier: AGPL-3.0-or-later
//! Matrix Power - Exponentiation by squaring (GPU implementation)
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Multi-pass iterative (log(n) matmuls)
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates square matrix
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::DeviceCapabilities;
use crate::error::{BarracudaError, Result};
use crate::ops::matmul::MatMul;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

pub struct MatrixPower {
    input: Tensor,
    power: i32,
}

impl MatrixPower {
    pub fn new(input: Tensor, power: i32) -> Result<Self> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "matrix_power",
                "Requires at least 2D tensor",
            ));
        }

        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];

        if rows != cols {
            return Err(BarracudaError::invalid_op(
                "matrix_power",
                format!("Requires square matrix, got {rows}x{cols}"),
            ));
        }

        if power < 0 {
            return Err(BarracudaError::invalid_op(
                "matrix_power",
                "Negative powers not supported (requires matrix inversion)",
            ));
        }

        Ok(Self { input, power })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/math/matrix_power_f64.wgsl"
            ))
        });
        &SHADER
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size = shape[shape.len() - 1];
        let matrix_size = size * size;

        if self.power == 0 {
            // Return identity matrix using WGSL shader
            let identity_buffer = device.create_buffer_f32(matrix_size)?;

            // Create parameters
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct MatrixPowerParams {
                size: u32,
                _pad1: u32,
                _pad2: u32,
                _pad3: u32,
            }

            let params = MatrixPowerParams {
                size: size as u32,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };

            let params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatrixPower Identity Params"),
                        contents: bytemuck::cast_slice(&[params]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

            // Compile shader
            let shader_module =
                device.compile_shader(Self::wgsl_shader(), Some("MatrixPower Shader"));

            // Create bind group layout
            let bind_group_layout =
                device
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("MatrixPower Identity Bind Group Layout"),
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
                                binding: 3,
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
                label: Some("MatrixPower Identity Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: identity_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create pipeline layout
            let pipeline_layout =
                device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("MatrixPower Identity Pipeline Layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            // Create pipeline
            let pipeline =
                device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("MatrixPower Identity Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: "init_identity",
                        cache: None,
                        compilation_options: Default::default(),
                    });

            // Encode and execute
            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatrixPower Identity Encoder"),
                    });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatrixPower Identity Pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch using standard 2D shader workgroup size (16, 16)
                let caps = DeviceCapabilities::from_device(device);
                let (workgroups_x, workgroups_y) = caps.dispatch_2d(size as u32, size as u32);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            device.submit_and_poll(Some(encoder.finish()));

            let output_shape = shape.to_vec();
            let output_elem_count = output_shape.iter().product::<usize>();
            let output_data =
                crate::utils::read_buffer(device, &identity_buffer, output_elem_count)?;
            return Ok(Tensor::new(output_data, output_shape, device.clone()));
        }

        if self.power == 1 {
            return Ok(self.input);
        }

        // Exponentiation by squaring: M^n
        // result = I, base = M; while n>0: if n odd then result *= base; base *= base; n /= 2
        let mut n = self.power as u32;
        let mut base = self.input.clone();
        let identity_data: Vec<f32> = (0..matrix_size)
            .map(|i| if i % (size + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut result = Tensor::from_data(&identity_data, shape.to_vec(), device.clone())?;

        while n > 0 {
            if n % 2 == 1 {
                result = MatMul::new(&result, &base).execute()?;
            }
            base = MatMul::new(&base, &base).execute()?;
            n /= 2;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_matrix_power_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let matrix = Tensor::from_vec_on(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = MatrixPower::new(matrix, 2).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();
        // (2I)^2 = 4I
        assert!((output[0] - 4.0).abs() < 1e-4);
        assert!((output[3] - 4.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_matrix_power_zero() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let matrix = Tensor::from_vec_on(vec![5.0, 3.0, 2.0, 1.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = MatrixPower::new(matrix, 0).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!(output[1].abs() < 1e-5);
        assert!(output[2].abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_matrix_power_one() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let matrix = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = MatrixPower::new(matrix.clone(), 1)
            .unwrap()
            .execute()
            .unwrap();
        let output = result.to_vec().unwrap();
        let input = matrix.to_vec().unwrap();
        assert_eq!(output, input);
    }
}
