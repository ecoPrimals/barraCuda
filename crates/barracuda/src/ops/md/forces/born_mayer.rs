// SPDX-License-Identifier: AGPL-3.0-or-later
//! Born-Mayer Force Calculation
//!
//! **Physics**: Hard-core repulsion (ionic crystals, close approach)
//! **Formula**: F = A/(ρ)·exp(-r/ρ)·r̂
//! **Use Case**: Ionic solids (`NaCl`), core-shell models, collisions
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ Per-particle parameters (agnostic)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Born-Mayer repulsive force calculation
///
/// Models hard-core repulsion between ions.
/// Exponential form prevents particle overlap in ionic crystals.
pub struct BornMayerForce {
    positions: Tensor,  // [N, 3]
    amplitudes: Tensor, // [N] - per-particle A
    ranges: Tensor,     // [N] - per-particle ρ
    cutoff_radius: f32,
}

impl BornMayerForce {
    /// Create a Born-Mayer force calculator with per-particle amplitude and range parameters.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if positions shape is not [N, 3], or amplitudes/ranges shape does not match.
    pub fn new(
        positions: Tensor,
        amplitudes: Tensor,
        ranges: Tensor,
        cutoff_radius: Option<f32>,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];

        // Validate amplitudes and ranges
        for tensor in [&amplitudes, &ranges] {
            let shape = tensor.shape();
            if shape.len() != 1 || shape[0] != n_particles {
                return Err(BarracudaError::InvalidShape {
                    expected: vec![n_particles],
                    actual: shape.to_vec(),
                });
            }
        }

        Ok(Self {
            positions,
            amplitudes,
            ranges,
            cutoff_radius: cutoff_radius.unwrap_or(5.0),
        })
    }

    /// Compute Born-Mayer repulsive forces for all particles and return the force tensor [N, 3].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        let output_size = (n_particles * 3 * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Born-Mayer Forces Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            cutoff_radius: f32,
            pad1: f32,
            pad2: f32,
        }

        let params = Params {
            n_particles: n_particles as u32,
            cutoff_radius: self.cutoff_radius,
            pad1: 0.0,
            pad2: 0.0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Born-Mayer Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "Born-Mayer")
            .shader(include_str!("born_mayer.wgsl"), "main")
            .storage_read(0, self.positions.buffer())
            .storage_read(1, self.amplitudes.buffer())
            .storage_read(2, self.ranges.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(n_particles as u32)
            .submit()?;

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
    async fn test_born_mayer_repulsion() {
        let device = crate::device::test_pool::get_test_device().await;

        // Two ions at close range (strong repulsion)
        let positions = vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0]; // Close!
        let amplitudes = vec![1000.0, 1000.0]; // Strong repulsion
        let ranges = vec![0.3, 0.3]; // Short range

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let amp_tensor = Tensor::from_data(&amplitudes, vec![2], device.clone()).unwrap();
        let range_tensor = Tensor::from_data(&ranges, vec![2], device).unwrap();

        let bm = BornMayerForce::new(pos_tensor, amp_tensor, range_tensor, None).unwrap();
        let forces = bm.execute().unwrap();

        assert_eq!(forces.shape(), &[2, 3]);
    }
}
