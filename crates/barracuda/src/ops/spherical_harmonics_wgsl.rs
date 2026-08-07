// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real spherical harmonics `Y_l^m(theta`, phi) for multipole expansion - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Real spherical harmonics `Y_l^m(theta`, phi).
/// `theta_phi`: interleaved [theta0, phi0, theta1, phi1, ...]
/// l: degree (0..6), m: order (can be negative)
pub struct SphericalHarmonics {
    theta_phi: Tensor,
    l: u32,
    m: i32,
}

impl SphericalHarmonics {
    /// Create spherical harmonics `Y_l^m` evaluation for (θ, φ) angle pairs.
    #[must_use]
    pub fn new(theta_phi: Tensor, l: u32, m: i32) -> Self {
        Self { theta_phi, l, m }
    }

    /// Evaluate `Y_l^m(θ`, φ) for all (θ, φ) pairs in the input.
    /// # Panics
    /// Panics if `theta_phi` has odd length (must be even for theta/phi pairs).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.theta_phi.device();
        let total_elements: usize = self.theta_phi.shape().iter().product();
        assert!(
            total_elements.is_multiple_of(2),
            "theta_phi must have even length (theta, phi pairs)"
        );
        let size = total_elements / 2;

        if size == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        let output_buffer = device.create_buffer_f32(size)?;

        let abs_m = self.m.unsigned_abs();
        let m_is_positive = u32::from(self.m > 0);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            l: u32,
            abs_m: u32,
            m_is_positive: u32,
        }

        let params = Params {
            size: size as u32,
            l: self.l,
            abs_m,
            m_is_positive,
        };
        let params_buffer = device.create_uniform_buffer("SphericalHarmonics Params", &params);

        ComputeDispatch::new(device, "SphericalHarmonics")
            .shader(
                include_str!("../shaders/special/spherical_harmonics.wgsl"),
                "main",
            )
            .storage_read(0, self.theta_phi.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![size],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute real spherical harmonic `Y_l^m` at (theta, phi) points.
    /// `theta_phi`: interleaved [theta0, phi0, theta1, phi1, ...]
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn spherical_harmonics(self, l: u32, m: i32) -> Result<Self> {
        SphericalHarmonics::new(self, l, m).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spherical_harmonics_y00() {
        let device = crate::device::test_pool::get_test_device().await;
        // Y_0^0 = 1/(2*sqrt(pi)) = 0.282094...
        let theta_phi = vec![0.0f32, 0.0, 1.0, 2.0]; // two points
        let input = Tensor::new(theta_phi, vec![4], device);
        let output = input.spherical_harmonics(0, 0).unwrap();
        let result = output.to_vec().unwrap();
        let expected = 0.5 / std::f32::consts::PI.sqrt(); // Y_0^0 = 1/sqrt(4*pi)
        assert!((result[0] - expected).abs() < 1e-5);
        assert!((result[1] - expected).abs() < 1e-5);
    }
}
