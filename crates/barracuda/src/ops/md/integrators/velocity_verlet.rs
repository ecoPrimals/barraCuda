// SPDX-License-Identifier: AGPL-3.0-or-later
//! Velocity-Verlet Time Integration
//!
//! **Physics**: Symplectic integrator for classical MD
//! **Properties**: Energy-conserving, time-reversible, 2nd-order accurate
//! **Use Case**: Molecular dynamics, planetary motion
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader  
//! - ✅ Zero unsafe code
//! - ✅ Agnostic (no hardcoded system)

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Velocity-Verlet time integration
///
/// Updates positions and velocities for one time step.
/// Requires forces at both t and t+Δt.
pub struct VelocityVerlet {
    positions: Tensor,  // [N, 3]
    velocities: Tensor, // [N, 3]
    forces_old: Tensor, // [N, 3] at time t
    forces_new: Tensor, // [N, 3] at time t+Δt
    masses: Tensor,     // [N]
    dt: f32,
}

impl VelocityVerlet {
    pub fn new(
        positions: Tensor,
        velocities: Tensor,
        forces_old: Tensor,
        forces_new: Tensor,
        masses: Tensor,
        dt: f32,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];

        // Validate all tensors have matching shapes
        for tensor in [&velocities, &forces_old, &forces_new] {
            if tensor.shape() != pos_shape {
                return Err(BarracudaError::InvalidShape {
                    expected: pos_shape.to_vec(),
                    actual: tensor.shape().to_vec(),
                });
            }
        }

        let mass_shape = masses.shape();
        if mass_shape.len() != 1 || mass_shape[0] != n_particles {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n_particles],
                actual: mass_shape.to_vec(),
            });
        }

        if dt <= 0.0 {
            return Err(BarracudaError::Device(
                "Time step dt must be positive".to_string(),
            ));
        }

        Ok(Self {
            positions,
            velocities,
            forces_old,
            forces_new,
            masses,
            dt,
        })
    }

    /// Execute Velocity-Verlet integration
    ///
    /// # Returns
    /// (positions_new, velocities_new) at time t+Δt
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        // Create output buffers
        let buffer_size = (n_particles * 3 * std::mem::size_of::<f32>()) as u64;

        let positions_new_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VV Positions New"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocities_new_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VV Velocities New"),
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
                label: Some("VV Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Velocity-Verlet Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("velocity_verlet.wgsl").into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("VV BGL"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
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
                    label: Some("VV PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VV Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VV BG"),
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
                    resource: self.forces_old.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.forces_new.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.masses.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: positions_new_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: velocities_new_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("VV Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VV Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

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

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_velocity_verlet_single_particle() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Single particle in free space with constant force
        let positions = vec![0.0, 0.0, 0.0];
        let velocities = vec![1.0, 0.0, 0.0]; // v_x = 1
        let forces_old = vec![2.0, 0.0, 0.0]; // F = 2
        let forces_new = vec![2.0, 0.0, 0.0]; // Constant force
        let masses = vec![1.0]; // m = 1
        let dt = 0.1;

        let pos_tensor = Tensor::from_data(&positions, vec![1, 3], device.clone()).unwrap();
        let vel_tensor = Tensor::from_data(&velocities, vec![1, 3], device.clone()).unwrap();
        let f_old_tensor = Tensor::from_data(&forces_old, vec![1, 3], device.clone()).unwrap();
        let f_new_tensor = Tensor::from_data(&forces_new, vec![1, 3], device.clone()).unwrap();
        let mass_tensor = Tensor::from_data(&masses, vec![1], device.clone()).unwrap();

        // Verify inputs are correct
        let pos_check = pos_tensor.to_vec().unwrap();
        let vel_check = vel_tensor.to_vec().unwrap();
        let f_old_check = f_old_tensor.to_vec().unwrap();
        let f_new_check = f_new_tensor.to_vec().unwrap();
        let mass_check = mass_tensor.to_vec().unwrap();

        println!("Input positions: {:?}", pos_check);
        println!("Input velocities: {:?}", vel_check);
        println!("Input forces_old: {:?}", f_old_check);
        println!("Input forces_new: {:?}", f_new_check);
        println!("Input masses: {:?}", mass_check);

        assert_eq!(pos_check, positions);
        assert_eq!(vel_check, velocities);
        assert_eq!(f_old_check, forces_old);
        assert_eq!(f_new_check, forces_new);
        assert_eq!(mass_check, masses);

        let vv = VelocityVerlet::new(
            pos_tensor,
            vel_tensor,
            f_old_tensor,
            f_new_tensor,
            mass_tensor,
            dt,
        )
        .unwrap();

        let (pos_new, vel_new) = vv.execute().unwrap();

        let pos_data = pos_new.to_vec().unwrap();
        let vel_data = vel_new.to_vec().unwrap();

        println!("pos_data: {:?}", pos_data);
        println!("vel_data: {:?}", vel_data);

        // Check physics: x = x0 + v*t + 0.5*a*t^2
        // x = 0 + 1*0.1 + 0.5*2*0.01 = 0.11
        let expected_x = 0.0 + 1.0 * dt + 0.5 * 2.0 * dt * dt;
        println!("Expected x: {}, got: {}", expected_x, pos_data[0]);
        assert!((pos_data[0] - expected_x).abs() < 1e-5, "Position update");

        // v = v0 + a*t
        // v = 1 + 2*0.1 = 1.2
        let expected_v = 1.0 + 2.0 * dt;
        println!("Expected v: {}, got: {}", expected_v, vel_data[0]);
        assert!((vel_data[0] - expected_v).abs() < 1e-5, "Velocity update");

        println!("✅ Velocity-Verlet validated");
    }
}
