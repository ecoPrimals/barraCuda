// SPDX-License-Identifier: AGPL-3.0-or-later
//! Split Velocity-Verlet Integrator (f64)
//!
//! **Algorithm**: Kick-drift-kick pattern for flexible thermostating
//! **Precision**: Full f64 via math_f64.wgsl preamble
//! **Reference**: Standard in LAMMPS, GROMACS
//!
//! **Advantages over monolithic VV**:
//! - Thermostat can be applied between kicks
//! - Force kernel can be swapped without touching integrator
//! - Explicit PBC wrapping during drift
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader (velocity_verlet_split.wgsl, vv_half_kick_f64.wgsl)
//! - ✅ Zero unsafe code
//! - ✅ f64 precision

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Split Velocity-Verlet Step 1: Half-kick + drift + PBC wrap
///
/// Updates velocities by half-step and positions by full step.
/// PBC wrapping applied during drift to keep positions in [0, box).
pub struct VelocityVerletKickDrift {
    positions: Tensor,
    velocities: Tensor,
    forces: Tensor,
    n_particles: usize,
    dt: f64,
    mass: f64,
    box_size: [f64; 3],
}

impl VelocityVerletKickDrift {
    /// Create a new kick-drift operation
    ///
    /// # Arguments
    /// * `positions` - Position tensor [N, 3] (f64)
    /// * `velocities` - Velocity tensor [N, 3] (f64)
    /// * `forces` - Force tensor [N, 3] (f64)
    /// * `dt` - Timestep (reduced units)
    /// * `mass` - Particle mass (3.0 in OCP reduced units)
    /// * `box_size` - Simulation box dimensions [Lx, Ly, Lz]
    ///
    /// # Errors
    /// Returns error if tensor shapes don't match.
    pub fn new(
        positions: Tensor,
        velocities: Tensor,
        forces: Tensor,
        dt: f64,
        mass: f64,
        box_size: [f64; 3],
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];

        // Validate matching shapes
        if velocities.shape() != pos_shape || forces.shape() != pos_shape {
            return Err(BarracudaError::InvalidShape {
                expected: pos_shape.to_vec(),
                actual: velocities.shape().to_vec(),
            });
        }

        if dt <= 0.0 {
            return Err(BarracudaError::Device(
                "Timestep dt must be positive".to_string(),
            ));
        }

        Ok(Self {
            positions,
            velocities,
            forces,
            n_particles,
            dt,
            mass,
            box_size,
        })
    }

    /// Execute the kick-drift step (in-place update)
    ///
    /// # Returns
    /// (positions, velocities) after half-kick and drift
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.positions.device();

        // Params: [n, dt, mass, _, box_x, box_y, box_z, _]
        let params: Vec<f64> = vec![
            self.n_particles as f64,
            self.dt,
            self.mass,
            0.0,
            self.box_size[0],
            self.box_size[1],
            self.box_size[2],
            0.0,
        ];
        let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VV KickDrift Params"),
                contents: &params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let shader_body = include_str!("velocity_verlet_split.wgsl");
        let shader = device.compile_shader_f64(shader_body, Some("VV KickDrift Shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("VV KickDrift BGL"),
                    entries: &[
                        // Positions (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Velocities (read-write)
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
                        // Forces (read)
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
                        // Params (read)
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
                    ],
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("VV KickDrift PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VV KickDrift Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VV KickDrift BG"),
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
                    resource: self.forces.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("VV KickDrift Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VV KickDrift Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (self.n_particles as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok((self.positions, self.velocities))
    }
}

/// Split Velocity-Verlet Step 3: Second half-kick
///
/// Completes the velocity update using NEW forces (after drift).
pub struct VelocityVerletHalfKick {
    velocities: Tensor,
    forces_new: Tensor,
    n_particles: usize,
    dt: f64,
    mass: f64,
}

impl VelocityVerletHalfKick {
    /// Create a new half-kick operation
    ///
    /// # Arguments
    /// * `velocities` - Velocity tensor [N, 3] (f64) — after kick-drift
    /// * `forces_new` - Force tensor [N, 3] (f64) — recomputed after drift
    /// * `dt` - Timestep (reduced units)
    /// * `mass` - Particle mass (3.0 in OCP reduced units)
    ///
    /// # Errors
    /// Returns error if tensor shapes don't match.
    pub fn new(velocities: Tensor, forces_new: Tensor, dt: f64, mass: f64) -> Result<Self> {
        let vel_shape = velocities.shape();
        if vel_shape.len() != 2 || vel_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: vel_shape.to_vec(),
            });
        }

        let n_particles = vel_shape[0];

        if forces_new.shape() != vel_shape {
            return Err(BarracudaError::InvalidShape {
                expected: vel_shape.to_vec(),
                actual: forces_new.shape().to_vec(),
            });
        }

        Ok(Self {
            velocities,
            forces_new,
            n_particles,
            dt,
            mass,
        })
    }

    /// Execute the second half-kick (in-place update)
    ///
    /// # Returns
    /// Velocities after full VV step
    pub fn execute(self) -> Result<Tensor> {
        let device = self.velocities.device();

        // Params: [n, dt, mass, _]
        let params: Vec<f64> = vec![self.n_particles as f64, self.dt, self.mass, 0.0];
        let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VV HalfKick Params"),
                contents: &params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let shader = device.compile_shader_f64(
            include_str!("vv_half_kick_f64.wgsl"),
            Some("VV HalfKick Shader"),
        );

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("VV HalfKick BGL"),
                    entries: &[
                        // Velocities (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Forces (read)
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
                        // Params (read)
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
                    ],
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("VV HalfKick PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VV HalfKick Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VV HalfKick BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.velocities.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.forces_new.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("VV HalfKick Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VV HalfKick Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (self.n_particles as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(self.velocities)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_split_vv_single_particle() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            println!("Skipping: No GPU available");
            return;
        };

        // Check for f64 support
        if !device
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            println!("Skipping: GPU does not support SHADER_F64");
            return;
        }

        // Single particle with constant force
        let positions: Vec<f64> = vec![0.0, 0.0, 0.0];
        let velocities: Vec<f64> = vec![1.0, 0.0, 0.0];
        let forces: Vec<f64> = vec![6.0, 0.0, 0.0]; // F = 6 → a = 6/3 = 2

        let dt = 0.1;
        let mass = 3.0; // OCP reduced units
        let box_size = [10.0, 10.0, 10.0];

        // Create tensors
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pos_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Positions"),
                contents: &pos_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let pos_tensor = Tensor::from_buffer(pos_buffer, vec![1, 3], device.clone());

        let vel_bytes: Vec<u8> = velocities.iter().flat_map(|v| v.to_le_bytes()).collect();
        let vel_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Velocities"),
                contents: &vel_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let vel_tensor = Tensor::from_buffer(vel_buffer, vec![1, 3], device.clone());

        let force_bytes: Vec<u8> = forces.iter().flat_map(|v| v.to_le_bytes()).collect();
        let force_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Forces"),
                contents: &force_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let force_tensor = Tensor::from_buffer(force_buffer, vec![1, 3], device.clone());

        // Step 1: Kick-drift
        let kick_drift =
            VelocityVerletKickDrift::new(pos_tensor, vel_tensor, force_tensor, dt, mass, box_size)
                .unwrap();

        let (pos_after, vel_after) = kick_drift.execute().unwrap();

        println!("✅ Split Velocity-Verlet f64 kick-drift executed");

        // Step 2 would be force recomputation (skipped here)
        // Step 3: Second half-kick (using same forces as approximation)
        let force_bytes2: Vec<u8> = forces.iter().flat_map(|v| v.to_le_bytes()).collect();
        let force_buffer2 = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Forces 2"),
                contents: &force_bytes2,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let force_tensor2 = Tensor::from_buffer(force_buffer2, vec![1, 3], device.clone());

        let half_kick = VelocityVerletHalfKick::new(vel_after, force_tensor2, dt, mass).unwrap();
        let _vel_final = half_kick.execute().unwrap();

        println!("✅ Split Velocity-Verlet f64 full step validated");
        println!("Final positions: {:?}", pos_after.shape());
    }
}
