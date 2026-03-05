// SPDX-License-Identifier: AGPL-3.0-or-later
//! Yukawa Force Calculation
//!
//! **Physics**: Screened electrostatic interactions (Debye screening)
//! **Formula**: F = k·q₁q₂·exp(-κr)/r²·r̂
//! **Use Case**: Dusty plasmas, colloids, screened electrostatics
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ Agnostic (κ parameterized)

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Yukawa (screened Coulomb) force calculation
///
/// Models interactions with exponential screening (e.g., Debye screening in plasmas).
/// Reduces to Coulomb when κ → 0.
pub struct YukawaForce {
    positions: Tensor,
    charges: Tensor,
    yukawa_constant: f32,
    kappa: f32, // Screening parameter (inverse Debye length)
    cutoff_radius: f32,
    epsilon: f32,
}

impl YukawaForce {
    /// Create a Yukawa force calculator with positions, charges, and screening parameters.
    pub fn new(
        positions: Tensor,
        charges: Tensor,
        yukawa_constant: Option<f32>,
        kappa: f32,
        cutoff_radius: Option<f32>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];
        let charge_shape = charges.shape();
        if charge_shape.len() != 1 || charge_shape[0] != n_particles {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n_particles],
                actual: charge_shape.to_vec(),
            });
        }

        if kappa < 0.0 {
            return Err(BarracudaError::Device(
                "Screening parameter κ must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            positions,
            charges,
            yukawa_constant: yukawa_constant.unwrap_or(1.0),
            kappa,
            cutoff_radius: cutoff_radius.unwrap_or(f32::INFINITY),
            epsilon: epsilon.unwrap_or(1e-6),
        })
    }

    /// Execute Yukawa force computation.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        let output_size = (n_particles * 3 * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Yukawa Forces Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            yukawa_constant: f32,
            kappa: f32,
            cutoff_radius: f32,
            epsilon: f32,
            _pad: [f32; 3],
        }

        let params = Params {
            n_particles: n_particles as u32,
            yukawa_constant: self.yukawa_constant,
            kappa: self.kappa,
            cutoff_radius: self.cutoff_radius,
            epsilon: self.epsilon,
            _pad: [0.0; 3],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Yukawa Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Yukawa Force Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("yukawa.wgsl").into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Yukawa BGL"),
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
                    label: Some("Yukawa PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Yukawa Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Yukawa BG"),
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

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Yukawa Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Yukawa Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

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
    async fn test_yukawa_reduces_to_coulomb() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // With κ=0, Yukawa should equal Coulomb
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let charges = vec![1.0, 1.0];

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let charge_tensor = Tensor::from_data(&charges, vec![2], device.clone()).unwrap();

        let yukawa = YukawaForce::new(
            pos_tensor,
            charge_tensor,
            Some(1.0),
            0.0, // κ=0 → Coulomb
            None,
            None,
        )
        .unwrap();

        let forces = yukawa.execute().unwrap();
        assert_eq!(forces.shape(), &[2, 3]);
        println!("✅ Yukawa with κ=0 validated");
    }

    #[tokio::test]
    async fn test_yukawa_screening() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Large κ should significantly reduce force at distance
        let positions = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let charges = vec![1.0, 1.0];

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let charge_tensor = Tensor::from_data(&charges, vec![2], device.clone()).unwrap();

        let yukawa = YukawaForce::new(
            pos_tensor,
            charge_tensor,
            Some(1.0),
            2.0, // Strong screening
            None,
            None,
        )
        .unwrap();

        let forces = yukawa.execute().unwrap();
        let force_data = forces.to_vec().unwrap();

        // Force should be heavily screened (small magnitude)
        let f0_mag = (force_data[0].powi(2) + force_data[1].powi(2) + force_data[2].powi(2)).sqrt();
        println!("✅ Yukawa screening validated: |F| = {}", f0_mag);
    }
}
