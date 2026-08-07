// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Polynomial Multiplication Operation
//!
//! **Purpose**: Multiply two FHE ciphertext polynomials on GPU
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ Hardware-agnostic (wgpu backend selection)
//! - ✅ Numerically precise (Barrett reduction for 128-bit)
//! - ✅ Production-ready (full error handling)
//! - ✅ Canonical pattern: Tensor inputs/outputs, device from runtime

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

/// FHE polynomial multiplication operation
///
/// Multiplies two polynomials coefficient-wise with modular reduction.
pub struct FhePolyMul {
    poly_a: Tensor,
    poly_b: Tensor,
    degree: u32,
    modulus: u64,
    barrett_mu: u64,
}

impl FhePolyMul {
    /// Create a new FHE polynomial multiplication operation
    /// # Errors
    /// Returns [`Err`] if polynomial lengths do not match `degree*2`, tensors are on different
    /// devices, or modulus is zero.
    pub fn new(poly_a: Tensor, poly_b: Tensor, degree: u32, modulus: u64) -> Result<Self> {
        let expected_size = (degree as usize) * 2;
        if poly_a.len() != expected_size {
            return Err(BarracudaError::Device(format!(
                "poly_a length {} doesn't match expected {} (degree {} * 2)",
                poly_a.len(),
                expected_size,
                degree
            )));
        }
        if poly_b.len() != expected_size {
            return Err(BarracudaError::Device(format!(
                "poly_b length {} doesn't match expected {} (degree {} * 2)",
                poly_b.len(),
                expected_size,
                degree
            )));
        }

        if !std::ptr::eq(poly_a.device().as_ref(), poly_b.device().as_ref()) {
            return Err(BarracudaError::Device(
                "poly_a and poly_b must be on the same device".to_string(),
            ));
        }

        if modulus == 0 {
            return Err(BarracudaError::Device(
                "Modulus must be non-zero".to_string(),
            ));
        }

        let barrett_mu = u64::MAX / modulus;

        Ok(Self {
            poly_a,
            poly_b,
            degree,
            modulus,
            barrett_mu,
        })
    }

    /// Execute polynomial multiplication on GPU
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.poly_a.device();

        let result_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FHE Result Buffer"),
            size: (self.degree as u64 * 2 * std::mem::size_of::<u32>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            degree: u32,
            modulus_lo: u32,
            modulus_hi: u32,
            mu_lo: u32,
            mu_hi: u32,
            _padding: [u32; 3],
        }

        let params = Params {
            degree: self.degree,
            modulus_lo: self.modulus as u32,
            modulus_hi: (self.modulus >> 32) as u32,
            mu_lo: self.barrett_mu as u32,
            mu_hi: (self.barrett_mu >> 32) as u32,
            _padding: [0; 3],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FHE Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "FHE Poly Mul")
            .shader(include_str!("fhe_poly_mul.wgsl"), "fhe_poly_mul")
            .storage_read(0, self.poly_a.buffer())
            .storage_read(1, self.poly_b.buffer())
            .storage_rw(2, &result_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree)
            .submit()?;

        Ok(Tensor::from_buffer(
            result_buffer,
            vec![self.degree as usize * 2],
            device.clone(),
        ))
    }
}

/// Helper: Create FHE polynomial tensor from u64 coefficients
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation or data upload fails (e.g. device lost).
pub async fn create_fhe_poly_tensor(
    poly: &[u64],
    device: Arc<crate::device::WgpuDevice>,
) -> Result<Tensor> {
    let poly_u32: Vec<u32> = poly
        .iter()
        .flat_map(|&val| vec![val as u32, (val >> 32) as u32])
        .collect();
    Tensor::from_data_pod(&poly_u32, vec![poly_u32.len()], device)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ops::fhe_poly_add::create_fhe_poly_tensor;

    #[tokio::test]
    async fn test_fhe_poly_mul_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let degree = 8;
        let modulus = 97;

        let poly_a_data = vec![2u64, 3, 4, 5, 6, 7, 8, 9];
        let poly_b_data = vec![5u64, 4, 3, 2, 10, 10, 10, 10];

        let poly_a = create_fhe_poly_tensor(&poly_a_data, device.clone())
            .await
            .unwrap();
        let poly_b = create_fhe_poly_tensor(&poly_b_data, device.clone())
            .await
            .unwrap();

        let op = FhePolyMul::new(poly_a, poly_b, degree, modulus).unwrap();
        let result_tensor = op.execute().unwrap();

        let size = result_tensor.len();
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Staging"),
            size: (size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_encoder_guarded(&Default::default());
        encoder.copy_buffer_to_buffer(
            result_tensor.buffer(),
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<u32>()) as u64,
        );
        device.submit_commands(Some(encoder.finish()));

        let result_u32: Vec<u32> = device.map_staging_buffer(&staging_buffer, size).unwrap();

        let result: Vec<u64> = result_u32
            .chunks(2)
            .map(|pair| (pair[0] as u64) | ((pair[1] as u64) << 32))
            .collect();

        let expected: Vec<u64> = vec![10, 12, 12, 10, 60, 70, 80, 90];
        assert_eq!(result, expected);
    }
}
