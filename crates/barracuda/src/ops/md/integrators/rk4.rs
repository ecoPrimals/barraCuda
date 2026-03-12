// SPDX-License-Identifier: AGPL-3.0-only
//! Runge-Kutta 4th Order Time Integration
//!
//! **Physics**: 4th-order accurate general ODE solver
//! **Properties**: Error ~ Δt⁵, self-starting
//! **Use Case**: Stiff ODEs, chemical kinetics, smooth dynamics
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader  
//! - ✅ Zero unsafe code

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// RK4 time integration
///
/// 4th-order accurate ODE solver for general dynamics.
/// For MD: treats acceleration as constant within timestep (simplified RK4).
pub struct Rk4 {
    positions: Tensor,     // [N, 3]
    velocities: Tensor,    // [N, 3]
    accelerations: Tensor, // [N, 3]
    dt: f32,
}

impl Rk4 {
    /// Creates an RK4 integrator with positions, velocities, accelerations, and time step.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if tensor shapes mismatch, or dt <= 0.
    pub fn new(
        positions: Tensor,
        velocities: Tensor,
        accelerations: Tensor,
        dt: f32,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        // Validate all tensors have matching shapes
        for tensor in [&velocities, &accelerations] {
            if tensor.shape() != pos_shape {
                return Err(BarracudaError::InvalidShape {
                    expected: pos_shape.to_vec(),
                    actual: tensor.shape().to_vec(),
                });
            }
        }

        if dt <= 0.0 {
            return Err(BarracudaError::Device(
                "Time step dt must be positive".to_string(),
            ));
        }

        Ok(Self {
            positions,
            velocities,
            accelerations,
            dt,
        })
    }

    /// Execute RK4 integration
    ///
    /// # Returns
    /// (`positions_new`, `velocities_new`) at time t+Δt
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        // Create output buffers
        let buffer_size = (n_particles * 3 * std::mem::size_of::<f32>()) as u64;

        let positions_new_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RK4 Positions New"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocities_new_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RK4 Velocities New"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            dt: f32,
            pad1: f32,
            pad2: f32,
        }

        let params = Params {
            n_particles: n_particles as u32,
            dt: self.dt,
            pad1: 0.0,
            pad2: 0.0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RK4 Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(include_str!("rk4.wgsl"), Some("RK4 Shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RK4 BGL"),
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
                    label: Some("RK4 PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RK4 Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RK4 BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.positions.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.velocities.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.accelerations.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: positions_new_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: velocities_new_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("RK4 Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RK4 Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let positions_new =
            Tensor::from_buffer(positions_new_buffer, vec![n_particles, 3], device.clone());

        let velocities_new =
            Tensor::from_buffer(velocities_new_buffer, vec![n_particles, 3], device.clone());

        Ok((positions_new, velocities_new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rk4_constant_acceleration() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Single particle with constant acceleration
        let positions = vec![0.0, 0.0, 0.0];
        let velocities = vec![1.0, 0.0, 0.0];
        let accelerations = vec![2.0, 0.0, 0.0];
        let dt = 0.1;

        let pos_tensor = Tensor::from_data(&positions, vec![1, 3], device.clone()).unwrap();
        let vel_tensor = Tensor::from_data(&velocities, vec![1, 3], device.clone()).unwrap();
        let acc_tensor = Tensor::from_data(&accelerations, vec![1, 3], device).unwrap();

        // Verify inputs
        assert_eq!(pos_tensor.to_vec().unwrap(), positions);
        assert_eq!(vel_tensor.to_vec().unwrap(), velocities);
        assert_eq!(acc_tensor.to_vec().unwrap(), accelerations);

        let rk4 = Rk4::new(pos_tensor, vel_tensor, acc_tensor, dt).unwrap();
        let (pos_new, vel_new) = rk4.execute().unwrap();

        let pos_data = pos_new.to_vec().unwrap();
        let vel_data = vel_new.to_vec().unwrap();

        println!("pos_data: {pos_data:?}");
        println!("vel_data: {vel_data:?}");

        // For constant acceleration, RK4 should be exact
        // x = x0 + v0*dt + 0.5*a*dt^2
        let expected_x = 0.0 + 1.0 * dt + 0.5 * 2.0 * dt * dt;
        // v = v0 + a*dt
        let expected_v = 1.0 + 2.0 * dt;

        println!("Expected x: {}, got: {}", expected_x, pos_data[0]);
        println!("Expected v: {}, got: {}", expected_v, vel_data[0]);

        assert!((pos_data[0] - expected_x).abs() < 1e-5, "Position update");
        assert!((vel_data[0] - expected_v).abs() < 1e-5, "Velocity update");

        println!("✅ RK4 validated");
    }
}
