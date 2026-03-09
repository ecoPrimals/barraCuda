// SPDX-License-Identifier: AGPL-3.0-or-later
//! Berendsen Thermostat
//!
//! **Physics**: Weak coupling to heat bath via velocity rescaling
//! **Formula**: v *= sqrt(1 + (dt/τ) * (`T_target/T_current` - 1))
//! **Use Case**: Equilibration phase only — does NOT sample canonical ensemble
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ f64 precision

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Berendsen velocity rescaling thermostat
///
/// The scale factor is computed on CPU from current temperature:
/// ```text
/// scale = sqrt(1 + (dt/tau) * (T_target/T_current - 1))
/// ```
/// Then applied uniformly to all velocities.
pub struct BerendsenThermostat {
    velocities: Tensor,
    scale_factor: f64,
}

impl BerendsenThermostat {
    /// Create a new Berendsen thermostat operation
    /// # Arguments
    /// * `velocities` - Velocity tensor [N, 3] (f64)
    /// * `scale_factor` - Pre-computed scale factor from temperature ratio
    /// # Errors
    /// Returns error if velocities tensor has wrong shape.
    pub fn new(velocities: Tensor, scale_factor: f64) -> Result<Self> {
        let shape = velocities.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: shape.to_vec(),
            });
        }

        Ok(Self {
            velocities,
            scale_factor,
        })
    }

    /// Compute the Berendsen scale factor
    /// # Arguments
    /// * `t_current` - Current temperature (reduced units)
    /// * `t_target` - Target temperature (reduced units)
    /// * `dt` - Timestep (reduced units)
    /// * `tau` - Coupling time constant (reduced units)
    #[must_use]
    pub fn compute_scale(t_current: f64, t_target: f64, dt: f64, tau: f64) -> f64 {
        if t_current < 1e-30 {
            return 1.0; // avoid division by zero
        }
        let ratio = 1.0 + (dt / tau) * (t_target / t_current - 1.0);
        ratio.max(0.0).sqrt()
    }

    /// Execute the thermostat (in-place velocity scaling)
    /// # Returns
    /// The same velocities tensor with scaled values
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.velocities.device();
        let n_particles = self.velocities.shape()[0];

        // Create params buffer: [n, scale, _, _]
        let params: Vec<f64> = vec![n_particles as f64, self.scale_factor, 0.0, 0.0];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Berendsen Params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let shader = device.compile_shader(
            include_str!("berendsen.wgsl"),
            Some("Berendsen Thermostat Shader"),
        );

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Berendsen BGL"),
                    entries: &[
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
                    ],
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Berendsen PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Berendsen Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Berendsen BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.velocities.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Berendsen Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Berendsen Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups = (n_particles as u32).div_ceil(WORKGROUP_SIZE_COMPACT);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(self.velocities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_berendsen_scale_computation() {
        // T_current = 0.01, T_target = 0.00633, dt = 0.01, tau = 0.05
        // Expected: scale = sqrt(1 + 0.2 * (0.633 - 1)) ≈ sqrt(0.9266) ≈ 0.9626
        let scale = BerendsenThermostat::compute_scale(0.01, 0.00633, 0.01, 0.05);
        assert!((scale - 0.9626).abs() < 0.01);
        println!("✅ Berendsen scale computation validated");
    }
}
