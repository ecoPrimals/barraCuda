//! FHE XOR Gate Operation
//!
//! **Purpose**: Perform Boolean XOR on FHE-encrypted data using GPU
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ Hardware-agnostic (wgpu backend selection)
//! - ✅ Numerically precise (modular arithmetic)
//! - ✅ Production-ready (full error handling)
//! - ✅ Canonical pattern: Tensor inputs/outputs, device from runtime

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

/// FHE XOR gate operation
///
/// Performs Boolean XOR on encrypted data using polynomial representation.
/// XOR(a,b) = a + b - 2*(a * b) mod q
pub struct FheXor {
    poly_a: Tensor,
    poly_b: Tensor,
    degree: u32,
    modulus: u64,
}

impl FheXor {
    /// Create a new FHE XOR gate operation
    pub fn new(poly_a: Tensor, poly_b: Tensor, degree: u32, modulus: u64) -> Result<Self> {
        if poly_a.len() != degree as usize || poly_b.len() != degree as usize {
            return Err(BarracudaError::Device(format!(
                "Polynomial length mismatch: expected {}, got {} and {}",
                degree,
                poly_a.len(),
                poly_b.len()
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

        Ok(Self {
            poly_a,
            poly_b,
            degree,
            modulus,
        })
    }

    /// Execute the XOR gate on two encrypted polynomials
    pub fn execute(self) -> Result<Tensor> {
        let device = self.poly_a.device();

        let result_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FHE XOR Output"),
            size: (self.degree as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            degree: u32,
            modulus_lo: u32,
            modulus_hi: u32,
            _pad: u32,
        }
        let params = Params {
            degree: self.degree,
            modulus_lo: (self.modulus & 0xFFFF_FFFF) as u32,
            modulus_hi: (self.modulus >> 32) as u32,
            _pad: 0,
        };
        let params_buffer = device.create_uniform_buffer("FHE XOR Params", &params);

        ComputeDispatch::new(device.as_ref(), "FHE XOR")
            .shader(include_str!("fhe_xor.wgsl"), "main")
            .storage_read(0, self.poly_a.buffer())
            .storage_read(1, self.poly_b.buffer())
            .storage_rw(2, &result_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree)
            .submit();

        Ok(Tensor::from_buffer(
            result_buffer,
            vec![self.degree as usize],
            device.clone(),
        ))
    }
}

/// Helper: Create FHE bit tensor from u64 coefficients (for bitwise ops)
pub async fn create_fhe_bit_tensor(
    poly: &[u64],
    device: Arc<crate::device::WgpuDevice>,
) -> Result<Tensor> {
    let poly_u32: Vec<u32> = poly.iter().map(|&x| x as u32).collect();
    Tensor::from_data_pod(&poly_u32, vec![poly_u32.len()], device)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    use crate::ops::fhe_and::create_fhe_bit_tensor;

    #[allow(unused_imports)]
    use wgpu::util::DeviceExt;

    #[tokio::test]
    async fn test_fhe_xor_same() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let poly_a_data = vec![1u64; 8];
        let poly_b_data = vec![1u64; 8];

        let poly_a = create_fhe_bit_tensor(&poly_a_data, device.clone())
            .await
            .unwrap();
        let poly_b = create_fhe_bit_tensor(&poly_b_data, device.clone())
            .await
            .unwrap();

        let op = FheXor::new(poly_a, poly_b, 8, 251).unwrap();
        let result_tensor = op.execute().unwrap();

        let size = result_tensor.len();
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Staging"),
            size: (size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            result_tensor.buffer(),
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<u32>()) as u64,
        );
        device.submit_and_poll(Some(encoder.finish()));

        let result_u32: Vec<u32> = device.map_staging_buffer(&staging_buffer, size).unwrap();

        assert_eq!(result_u32.len(), 8);
        assert!(result_u32.iter().all(|&x| x == 0), "1 XOR 1 should equal 0");
    }
}
