// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Modulus Switching Operation
//!
//! **Purpose**: Reduce ciphertext noise by switching to a smaller modulus
//!
//! **Algorithm**: Scale-and-round modulus reduction
//! - Noise reduction: ~log(q_old/q_new) bits
//! - Preserves plaintext (decrypt correctness maintained)
//! - Essential for leveled FHE schemes (BFV, BGV)
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ GPU-accelerated via compute shaders
//! - ✅ Numerically precise (exact rounding)
//! - ✅ Production-ready (full error handling)
//!
//! ## Mathematical Background
//!
//! Modulus switching converts a ciphertext under modulus q_old to modulus q_new:
//! ```text
//! ct_new = round((q_new / q_old) * ct_old) mod q_new
//! ```
//!
//! **Key Properties**:
//! 1. **Correctness**: Dec(ct_new, sk, q_new) = Dec(ct_old, sk, q_old)
//! 2. **Noise Reduction**: noise_new ≈ noise_old * (q_new / q_old)
//! 3. **Homomorphism**: Can continue operations under q_new
//!
//! **Use Cases**:
//! - **Noise Management**: Reduce accumulated noise before overflow
//! - **Leveled FHE**: Enable deeper circuits without bootstrapping
//! - **Bandwidth**: Smaller ciphertexts for network transmission
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use barracuda::ops::fhe_modulus_switch::FheModulusSwitch;
//! use barracuda::prelude::Tensor;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Ciphertext under large modulus (e.g., after multiplication)
//! let ct = Tensor::from_u64_poly(&ciphertext, degree).await?;
//!
//! // Switch to smaller modulus for noise reduction
//! let q_old = 1152921504606584833u64; // 60-bit prime
//! let q_new = 288230376151711777u64;  // 58-bit prime (4x smaller)
//!
//! let switch_op = FheModulusSwitch::new(ct, degree, q_old, q_new)?;
//! let ct_new = switch_op.execute()?;
//!
//! // ct_new has same plaintext, but ~4x less noise
//! # Ok(())
//! # }
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// FHE Modulus Switching operation
///
/// Scales ciphertext coefficients to smaller modulus while preserving plaintext.
pub struct FheModulusSwitch {
    input: Tensor,
    degree: u32,
    modulus_old: u64,
    modulus_new: u64,
}

impl FheModulusSwitch {
    /// Create a new modulus switching operation
    ///
    /// **Parameters**:
    /// - `input`: Ciphertext polynomial (2*degree u32 values, u64 emulated)
    /// - `degree`: Polynomial degree (power of 2)
    /// - `modulus_old`: Current modulus
    /// - `modulus_new`: Target modulus (must be < modulus_old)
    ///
    /// **Returns**: FheModulusSwitch operation ready to execute
    ///
    /// **Errors**:
    /// - Invalid degree (not power of 2)
    /// - modulus_new >= modulus_old
    /// - Input tensor size mismatch
    pub fn new(input: Tensor, degree: u32, modulus_old: u64, modulus_new: u64) -> Result<Self> {
        // ✅ VALIDATION: Degree must be power of 2
        if !degree.is_power_of_two() || degree < 4 {
            return Err(BarracudaError::InvalidInput {
                message: format!("Degree must be power of 2 >= 4, got {degree}"),
            });
        }

        // ✅ VALIDATION: New modulus must be smaller
        if modulus_new >= modulus_old {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "New modulus ({modulus_new}) must be < old modulus ({modulus_old})"
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
            modulus_old,
            modulus_new,
        })
    }

    /// Execute modulus switching on GPU
    ///
    /// **Returns**: Tensor with coefficients scaled to new modulus
    ///
    /// **Performance**: O(n) GPU parallel execution
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Create output buffer
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FHE Modulus Switch Output"),
            size: self.input.buffer().size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SwitchParams {
            degree: u32,
            modulus_old_lo: u32,
            modulus_old_hi: u32,
            modulus_new_lo: u32,
            modulus_new_hi: u32,
            _padding: [u32; 3],
        }
        let params = SwitchParams {
            degree: self.degree,
            modulus_old_lo: (self.modulus_old & 0xFFFF_FFFF) as u32,
            modulus_old_hi: (self.modulus_old >> 32) as u32,
            modulus_new_lo: (self.modulus_new & 0xFFFF_FFFF) as u32,
            modulus_new_hi: (self.modulus_new >> 32) as u32,
            _padding: [0; 3],
        };
        let params_buffer = device.create_uniform_buffer("FHE Modulus Switch Params", &params);

        ComputeDispatch::new(device.as_ref(), "FHE Modulus Switch")
            .shader(include_str!("fhe_modulus_switch.wgsl"), "modulus_switch")
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
    async fn test_modulus_switch_validation() {
        // Test invalid degree
        let result = FheModulusSwitch::new(
            Tensor::zeros(vec![8]).await.unwrap(),
            3, // Not power of 2
            12_289,
            6145,
        );
        assert!(result.is_err());

        // Test new modulus >= old modulus
        let result = FheModulusSwitch::new(
            Tensor::zeros(vec![8]).await.unwrap(),
            4,
            12_289,
            12_289, // Equal (should fail)
        );
        assert!(result.is_err());
    }

    // NOTE: Full integration tests require GPU + encryption setup
    // See examples/fhe_modulus_switch_validation.rs for round-trip tests
}
