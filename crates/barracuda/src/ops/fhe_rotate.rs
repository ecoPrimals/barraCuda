//! FHE Rotation Operation (CKKS Scheme)
//!
//! **Purpose**: Rotate ciphertext slots for SIMD operations
//!
//! **Algorithm**: Galois automorphism + key switching
//! - Apply automorphism: X → X^(2k+1) for rotation by k slots
//! - Re-linearize using rotation key (Galois key)
//! - Enables cyclic permutations of encrypted data
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ GPU-accelerated (parallel automorphism)
//! - ✅ Numerically precise (exact permutation)
//! - ✅ Production-ready (bounds checking)
//!
//! ## Mathematical Background
//!
//! In CKKS homomorphic encryption, plaintexts are encoded as:
//! ```text
//! m = (m_0, m_1, ..., m_{N/2-1}) in N/2 slots
//! ```
//!
//! Rotation by k slots applies Galois automorphism:
//! ```text
//! σ_k: X → X^(2k+1) mod (X^N + 1)
//! ```
//!
//! This cyclically permutes the slots:
//! ```text
//! rotate(m, k) = (m_k, m_{k+1}, ..., m_{N/2-1}, m_0, ..., m_{k-1})
//! ```
//!
//! ## Use Cases
//!
//! 1. **Vector Operations**: Sum all encrypted values (log N rotations)
//! 2. **Matrix Multiplication**: Row/column alignment
//! 3. **Neural Networks**: Convolutional operations on encrypted data
//! 4. **Signal Processing**: Circular convolutions
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use barracuda::ops::fhe_rotate::FheRotate;
//! use barracuda::prelude::Tensor;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // CKKS ciphertext (degree 4096)
//! let ct = Tensor::from_u64_poly(&ciphertext, 4096).await?;
//!
//! // Rotate by 5 slots to the left
//! let rotate_op = FheRotate::new(ct, 4096, 5, modulus)?;
//! let ct_rotated = rotate_op.execute()?;
//!
//! // Decrypt to verify: slots are cyclically shifted
//! # Ok(())
//! # }
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// FHE Rotation operation for CKKS scheme
///
/// Applies Galois automorphism to rotate encrypted slots.
pub struct FheRotate {
    input: Tensor,
    degree: u32,
    rotation: i32, // Can be negative for right rotation
    modulus: u64,
}

impl FheRotate {
    /// Create a new rotation operation
    ///
    /// **Parameters**:
    /// - `input`: Ciphertext polynomial (2*degree u32 values, u64 emulated)
    /// - `degree`: Polynomial degree (power of 2)
    /// - `rotation`: Number of slots to rotate (positive=left, negative=right)
    /// - `modulus`: Ciphertext modulus
    ///
    /// **Returns**: FheRotate operation ready to execute
    ///
    /// **Errors**:
    /// - Invalid degree (not power of 2)
    /// - Rotation out of valid range
    /// - Input tensor size mismatch
    ///
    /// **Note**: This is a simplified version. Full CKKS rotation requires
    /// rotation keys (Galois keys) for key switching after automorphism.
    /// This implementation applies the automorphism only.
    pub fn new(input: Tensor, degree: u32, rotation: i32, modulus: u64) -> Result<Self> {
        // ✅ VALIDATION: Degree must be power of 2
        if !degree.is_power_of_two() || degree < 4 {
            return Err(BarracudaError::InvalidInput {
                message: format!("Degree must be power of 2 >= 4, got {degree}"),
            });
        }

        // ✅ VALIDATION: Rotation must be in valid range
        let max_rotation = (degree / 2) as i32;
        if rotation.abs() > max_rotation {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Rotation {rotation} out of range [-{max_rotation}, {max_rotation}] for degree {degree}"
                ),
            });
        }

        // ✅ VALIDATION: Input tensor must be 2*degree (u64 as 2xu32)
        let expected_size = (degree * 2) as usize;
        if input.shape()[0] != expected_size {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Input must have {} elements (degree={}, u64 emulated), got {}",
                    expected_size,
                    degree,
                    input.shape()[0]
                ),
            });
        }

        Ok(Self {
            input,
            degree,
            rotation,
            modulus,
        })
    }

    /// Execute rotation on GPU
    ///
    /// **Returns**: Tensor with rotated coefficients (automorphism applied)
    ///
    /// **Performance**: O(n) GPU parallel execution
    ///
    /// **Note**: For complete CKKS rotation, apply key switching after this step
    /// using the rotation key corresponding to this rotation index.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Create output buffer
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FHE Rotate Output"),
            size: self.input.buffer().size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Normalize rotation to positive range [0, degree)
        let rotation_normalized = if self.rotation < 0 {
            (self.degree as i32 + self.rotation) as u32
        } else {
            self.rotation as u32
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RotateParams {
            degree: u32,
            rotation: u32,
            modulus_lo: u32,
            modulus_hi: u32,
        }
        let params = RotateParams {
            degree: self.degree,
            rotation: rotation_normalized,
            modulus_lo: (self.modulus & 0xFFFF_FFFF) as u32,
            modulus_hi: (self.modulus >> 32) as u32,
        };
        let params_buffer = device.create_uniform_buffer("FHE Rotate Params", &params);

        ComputeDispatch::new(device.as_ref(), "FHE Rotate")
            .shader(include_str!("fhe_rotate.wgsl"), "rotate_automorphism")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(self.degree)
            .submit();

        // Return result tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rotate_validation() {
        // Test invalid degree
        let result = FheRotate::new(Tensor::zeros(vec![8]).await.unwrap(), 3, 1, 12_289);
        assert!(result.is_err());

        // Test rotation out of range
        let result = FheRotate::new(Tensor::zeros(vec![8]).await.unwrap(), 4, 3, 12_289);
        assert!(result.is_err()); // Max rotation for degree 4 is 2

        // Test negative rotation out of range
        let result = FheRotate::new(Tensor::zeros(vec![8]).await.unwrap(), 4, -3, 12289);
        assert!(result.is_err());
    }

    // NOTE: Full integration tests require GPU + CKKS setup
    // See examples/fhe_rotate_validation.rs for round-trip tests
}
