// SPDX-License-Identifier: AGPL-3.0-only
//! Per-Particle Kinetic Energy Computation
//!
//! **Physics**: KE = 0.5 * m * v² per particle
//! **Use Case**: Temperature calculation: T = 2*`KE_total` / (3*N*`k_B`)
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ f64 precision

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Per-particle kinetic energy computation
///
/// Computes `KE_i` = 0.5 * mass * (vx² + vy² + vz²) for each particle.
/// Sum the output to get total kinetic energy.
pub struct KineticEnergy {
    velocities: Tensor,
    mass: f64,
}

impl KineticEnergy {
    /// Create a kinetic energy computation
    /// # Arguments
    /// * `velocities` - Velocity tensor [N, 3] (f64)
    /// * `mass` - Particle mass (reduced units)
    /// # Errors
    /// Returns error if velocities tensor has wrong shape.
    pub fn new(velocities: Tensor, mass: f64) -> Result<Self> {
        let shape = velocities.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: shape.to_vec(),
            });
        }

        Ok(Self { velocities, mass })
    }

    /// Execute the kinetic energy computation
    /// # Returns
    /// A tensor [N] with per-particle kinetic energy
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.velocities.device();
        let n_particles = self.velocities.shape()[0];

        // Output buffer for KE per particle
        let output_size = (n_particles * std::mem::size_of::<f64>()) as u64;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Kinetic Energy Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params buffer: [n, mass, _, _]
        let params: Vec<f64> = vec![n_particles as f64, self.mass, 0.0, 0.0];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KE Params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let shader = device.compile_shader(
            include_str!("kinetic_energy.wgsl"),
            Some("Kinetic Energy Shader"),
        );

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("KE BGL"),
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
                    label: Some("KE PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("KE Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("KE BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.velocities.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("KE Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("KE Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups = (n_particles as u32).div_ceil(WORKGROUP_SIZE_COMPACT);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![n_particles],
            device.clone(),
        ))
    }

    /// Compute temperature from total kinetic energy
    /// T* = 2 * `KE_total` / (3 * N) in reduced units (`k_B` = 1)
    #[must_use]
    pub fn temperature_from_ke(ke_total: f64, n_particles: usize) -> f64 {
        2.0 * ke_total / (3.0 * n_particles as f64)
    }
}
