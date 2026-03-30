// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coulomb Force Calculation
//!
//! **Physics**: Electrostatic interactions between charged particles
//! **Formula**: F = k * `q_i` * `q_j` / r² * r̂
//! **Use Case**: Ions, proteins, charged molecules
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ Capability-based dispatch
//! - ✅ Agnostic (no hardcoded constants)

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Coulomb force calculation operation
///
/// Computes electrostatic forces between all particle pairs.
/// Uses softened potential to avoid singularities.
pub struct CoulombForce {
    positions: Tensor,     // [N, 3]
    charges: Tensor,       // [N]
    coulomb_constant: f32, // k (can absorb units into charges)
    cutoff_radius: f32,    // Maximum interaction distance
    epsilon: f32,          // Softening parameter
}

impl CoulombForce {
    /// Create new Coulomb force calculation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if positions shape is not [N, 3], or charges shape does not match.
    ///
    /// # Arguments
    /// * `positions` - Particle positions [N, 3]
    /// * `charges` - Particle charges [N]
    /// * `coulomb_constant` - Coulomb constant k (default: 1.0)
    /// * `cutoff_radius` - Cutoff distance (default: infinity)
    /// * `epsilon` - Softening parameter (default: 1e-6)
    pub fn new(
        positions: Tensor,
        charges: Tensor,
        coulomb_constant: Option<f32>,
        cutoff_radius: Option<f32>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        // Validate positions shape [N, 3]
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];

        // Validate charges shape [N]
        let charge_shape = charges.shape();
        if charge_shape.len() != 1 || charge_shape[0] != n_particles {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n_particles],
                actual: charge_shape.to_vec(),
            });
        }

        Ok(Self {
            positions,
            charges,
            coulomb_constant: coulomb_constant.unwrap_or(1.0),
            cutoff_radius: cutoff_radius.unwrap_or(f32::INFINITY),
            epsilon: epsilon.unwrap_or(1e-6),
        })
    }

    /// Execute Coulomb force calculation
    ///
    /// # Returns
    /// Force tensor [N, 3] containing force vectors for each particle
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        // Create output buffer [N, 3]
        let output_size = (n_particles * 3 * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Coulomb Forces Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            coulomb_constant: f32,
            cutoff_radius: f32,
            epsilon: f32,
        }

        let params = Params {
            n_particles: n_particles as u32,
            coulomb_constant: self.coulomb_constant,
            cutoff_radius: self.cutoff_radius,
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Coulomb Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Load shader
        let shader =
            device.compile_shader(include_str!("coulomb.wgsl"), Some("Coulomb Force Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Coulomb BGL"),
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
                    label: Some("Coulomb PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Coulomb Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Coulomb BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.positions.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.charges.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Coulomb Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Coulomb Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // One workgroup per particle
            let workgroups = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![n_particles, 3],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coulomb_force_two_particles() {
        let device = crate::device::test_pool::get_test_device().await;

        // Two particles on x-axis
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let charges = vec![1.0, 1.0]; // Same sign = repulsion

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let charge_tensor = Tensor::from_data(&charges, vec![2], device).unwrap();

        // First verify input tensors are correct
        let pos_check = pos_tensor.to_vec().unwrap();
        let charge_check = charge_tensor.to_vec().unwrap();
        println!("Input positions: {pos_check:?}");
        println!("Input charges: {charge_check:?}");
        assert_eq!(pos_check, positions);
        assert_eq!(charge_check, charges);

        let coulomb = CoulombForce::new(pos_tensor, charge_tensor, None, None, None).unwrap();
        let forces = coulomb.execute().unwrap();

        let force_data = forces.to_vec().unwrap();
        println!("Output forces: {force_data:?}");

        // Force on particle 0 should be negative x (repulsion away from particle 1)
        // With physics direction fixed, force should point in -x
        assert!(
            force_data[0] < 0.0,
            "Particle 0 repelled in -x direction: got {}",
            force_data[0]
        );

        // Force on particle 1 should be positive x
        assert!(
            force_data[3] > 0.0,
            "Particle 1 repelled in +x direction: got {}",
            force_data[3]
        );

        // Newton's third law: F_0 = -F_1
        let f0 = force_data[0];
        let f1 = force_data[3];
        assert!((f0 + f1).abs() < 1e-4, "Newton's third law");

        println!("✅ Coulomb force validated");
    }

    #[tokio::test]
    async fn test_coulomb_force_opposite_charges() {
        let device = crate::device::test_pool::get_test_device().await;

        // Two particles with opposite charges
        let positions = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let charges = vec![1.0, -1.0]; // Opposite signs = attraction

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let charge_tensor = Tensor::from_data(&charges, vec![2], device).unwrap();

        // Verify inputs
        let pos_check = pos_tensor.to_vec().unwrap();
        let charge_check = charge_tensor.to_vec().unwrap();
        println!("Input positions: {pos_check:?}");
        println!("Input charges: {charge_check:?}");
        assert_eq!(pos_check, positions);
        assert_eq!(charge_check, charges);

        // Explicitly set large cutoff instead of INFINITY
        let coulomb = CoulombForce::new(
            pos_tensor,
            charge_tensor,
            Some(1.0),  // k = 1
            Some(10.0), // Explicit cutoff instead of INFINITY
            Some(1e-6), // epsilon
        )
        .unwrap();
        let forces = coulomb.execute().unwrap();

        let force_data = forces.to_vec().unwrap();
        println!("Output forces: {force_data:?}");

        // Force on particle 0 should be positive x (attracted toward particle 1)
        println!("Expected: force[0] > 0 (attraction in +x)");
        println!("Got: force[0] = {}", force_data[0]);
        assert!(
            force_data[0] > 0.0,
            "Particle 0 attracted in +x direction: got {}",
            force_data[0]
        );

        // Force on particle 1 should be negative x
        println!("Expected: force[3] < 0 (attraction in -x)");
        println!("Got: force[3] = {}", force_data[3]);
        assert!(
            force_data[3] < 0.0,
            "Particle 1 attracted in -x direction: got {}",
            force_data[3]
        );

        println!("✅ Coulomb attraction validated");
    }
}
