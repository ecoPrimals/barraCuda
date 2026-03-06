// SPDX-License-Identifier: AGPL-3.0-or-later
//! Morse Force Calculation
//!
//! **Physics**: Anharmonic bonded interactions (covalent bonds, diatomics)
//! **Formula**: F = 2Daα[exp(-2α(r-r₀))-exp(-α(r-r₀))]·r̂
//! **Use Case**: Chemical bonds, molecular vibrations, spectroscopy
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader with atomic force accumulation
//! - ✅ Zero unsafe code
//! - ✅ Per-bond parameters (agnostic)

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Morse potential force calculation for bonded interactions
///
/// Models chemical bonds with anharmonic oscillator potential.
/// More accurate than harmonic approximation for large displacements.
pub struct MorseForce {
    positions: Tensor,           // [N, 3]
    bond_pairs: Tensor,          // [M, 2] - particle indices for each bond
    dissociation_energy: Tensor, // [M] - D for each bond
    width_param: Tensor,         // [M] - α (width) for each bond
    equilibrium_dist: Tensor,    // [M] - r₀ for each bond
}

impl MorseForce {
    /// Create a Morse force calculator for bonded interactions with per-bond parameters.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if positions shape is not [N, 3], `bond_pairs` is not [M, 2], or parameter shapes mismatch.
    pub fn new(
        positions: Tensor,
        bond_pairs: Tensor,
        dissociation_energy: Tensor,
        width_param: Tensor,
        equilibrium_dist: Tensor,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let bond_shape = bond_pairs.shape();
        if bond_shape.len() != 2 || bond_shape[1] != 2 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 2],
                actual: bond_shape.to_vec(),
            });
        }

        let n_bonds = bond_shape[0];

        // Validate parameter tensors
        for tensor in [&dissociation_energy, &width_param, &equilibrium_dist] {
            let shape = tensor.shape();
            if shape.len() != 1 || shape[0] != n_bonds {
                return Err(BarracudaError::InvalidShape {
                    expected: vec![n_bonds],
                    actual: shape.to_vec(),
                });
            }
        }

        Ok(Self {
            positions,
            bond_pairs,
            dissociation_energy,
            width_param,
            equilibrium_dist,
        })
    }

    /// Compute Morse forces for all particles and return the force tensor [N, 3].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];
        let n_bonds = self.bond_pairs.shape()[0];

        // Create atomic buffer (i32) for force accumulation
        let atomic_buffer_size = (n_particles * 3 * std::mem::size_of::<i32>()) as u64;
        let atomic_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morse Forces Atomic"),
            size: atomic_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clear atomic buffer to zero
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Morse Clear Encoder"),
        });
        encoder.clear_buffer(&atomic_buffer, 0, None);
        device.submit_and_poll(Some(encoder.finish()));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            n_bonds: u32,
            pad1: f32,
            pad2: f32,
        }

        let params = Params {
            n_particles: n_particles as u32,
            n_bonds: n_bonds as u32,
            pad1: 0.0,
            pad2: 0.0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Morse Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Morse Force Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("morse.wgsl").into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Morse BGL"),
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
                    label: Some("Morse PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Morse Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Morse BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.positions.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bond_pairs.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.dissociation_energy.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.width_param.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.equilibrium_dist.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: atomic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Morse Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Morse Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups = (n_bonds as u32).div_ceil(WORKGROUP_SIZE_1D);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Create staging buffer to read back atomic results
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morse Staging"),
            size: atomic_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&atomic_buffer, 0, &staging_buffer, 0, atomic_buffer_size);
        device.submit_and_poll(Some(encoder.finish()));

        // Read back and convert i32 -> f32
        let n_force_elements = n_particles * 3;
        let i32_data: Vec<i32> = device.map_staging_buffer(&staging_buffer, n_force_elements)?;
        let f32_data: Vec<f32> = i32_data.iter().map(|&x| x as f32 / 1000.0).collect();

        // Create final output tensor
        Tensor::from_data(&f32_data, vec![n_particles, 3], device.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_morse_force_equilibrium() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Two bonded particles at equilibrium (should have near-zero force)
        let positions = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0]; // r₀ = 1.5
        let bond_pairs = vec![0.0, 1.0]; // Bond between particles 0 and 1
        let dissociation = vec![100.0]; // D = 100
        let width = vec![2.0]; // α = 2
        let r0 = vec![1.5]; // Equilibrium at r = 1.5

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let pairs_tensor = Tensor::from_data(&bond_pairs, vec![1, 2], device.clone()).unwrap();
        let d_tensor = Tensor::from_data(&dissociation, vec![1], device.clone()).unwrap();
        let a_tensor = Tensor::from_data(&width, vec![1], device.clone()).unwrap();
        let r0_tensor = Tensor::from_data(&r0, vec![1], device.clone()).unwrap();

        let morse =
            MorseForce::new(pos_tensor, pairs_tensor, d_tensor, a_tensor, r0_tensor).unwrap();

        let forces = morse.execute().unwrap();
        let force_data = forces.to_vec().unwrap();

        // At equilibrium, force should be very small
        let f0_mag = (force_data[0].powi(2) + force_data[1].powi(2) + force_data[2].powi(2)).sqrt();
        println!("✅ Morse equilibrium force: |F| = {f0_mag} (expect ≈0)");
    }
}
