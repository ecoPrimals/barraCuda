// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Polynomial Subtraction Operation
//!
//! **Purpose**: Subtract two FHE ciphertext polynomials on GPU
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ Hardware-agnostic (wgpu backend selection)
//! - ✅ Numerically precise (modular subtraction)
//! - ✅ Production-ready (full error handling)
//! - ✅ Canonical pattern: Tensor inputs/outputs, device from runtime

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

/// FHE polynomial subtraction operation
///
/// Subtracts two polynomials coefficient-wise with modular reduction.
///
/// ## Mathematical Operation
///
/// Given polynomials a(X) and b(X) over `Z_q`[X]/(X^N + 1):
/// ```text
/// result(X) = a(X) - b(X) mod q
/// ```
///
/// Where each coefficient is reduced modulo q.
pub struct FhePolySub {
    poly_a: Tensor,
    poly_b: Tensor,
    degree: u32,
    modulus: u64,
}

impl FhePolySub {
    /// Create a new FHE polynomial subtraction operation
    /// ## Parameters
    /// - `poly_a`: First polynomial tensor (u32 pairs representing u64 coefficients)
    /// - `poly_b`: Second polynomial tensor (u32 pairs representing u64 coefficients)
    /// - `degree`: Polynomial degree (N), typically 2048, 4096, or 8192
    /// - `modulus`: Modulus q (large prime, e.g., 2^60)
    /// # Errors
    /// Returns [`Err`] if polynomial lengths do not match `degree*2`, tensors are on different
    /// devices, or modulus is zero.
    pub fn new(poly_a: Tensor, poly_b: Tensor, degree: u32, modulus: u64) -> Result<Self> {
        // Validate inputs
        let expected_size = (degree as usize) * 2; // u32 pairs for u64
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

        // Ensure both tensors are on same device
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

        Ok(Self {
            poly_a,
            poly_b,
            degree,
            modulus,
        })
    }

    /// Execute polynomial subtraction on GPU
    /// ## Returns
    /// Result tensor: (`poly_a` - `poly_b`) mod q
    /// Data stays on GPU (no CPU readback)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.poly_a.device();

        // Create output buffer (u32 pairs for u64 coefficients)
        let result_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FHE Result Buffer"),
            size: (self.degree as u64 * 2 * std::mem::size_of::<u32>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            degree: u32,
            modulus_lo: u32,
            modulus_hi: u32,
            _padding: [u32; 5],
        }

        let params = Params {
            degree: self.degree,
            modulus_lo: self.modulus as u32,
            modulus_hi: (self.modulus >> 32) as u32,
            _padding: [0; 5],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FHE Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "FHE Poly Sub")
            .shader(include_str!("fhe_poly_sub.wgsl"), "fhe_poly_sub")
            .storage_read(0, self.poly_a.buffer())
            .storage_read(1, self.poly_b.buffer())
            .storage_rw(2, &result_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree)
            .submit()?;

        // Return tensor (data stays on GPU)
        Ok(Tensor::from_buffer(
            result_buffer,
            vec![self.degree as usize * 2], // u32 pairs
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
    async fn test_fhe_poly_sub_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let degree = 8;
        let modulus = 97;

        let poly_a_data = vec![50u64, 60, 70, 80, 90, 85, 75, 65];
        let poly_b_data = vec![10u64, 20, 30, 40, 50, 60, 70, 80];

        let poly_a = create_fhe_poly_tensor(&poly_a_data, device.clone())
            .await
            .unwrap();
        let poly_b = create_fhe_poly_tensor(&poly_b_data, device.clone())
            .await
            .unwrap();

        let op = FhePolySub::new(poly_a, poly_b, degree, modulus).unwrap();
        let result_tensor = op.execute().unwrap();

        // Read back for testing
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

        let expected: Vec<u64> = vec![40, 40, 40, 40, 40, 25, 5, 82];
        assert_eq!(result, expected);
    }
}
