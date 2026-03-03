use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

/// Point-wise multiplication of two polynomials in NTT domain
///
/// Given two polynomials A and B in NTT domain (already transformed),
/// computes C = A ⊙ B (element-wise product).
///
/// This is the core operation in fast polynomial multiplication:
///   poly_mul(a, b) = INTT(pointwise_mul(NTT(a), NTT(b)))
///
/// Complexity: O(N) - much faster than O(N²) convolution!
///
/// # Example
/// ```ignore
/// use barracuda::ops::{FheNtt, FhePointwiseMul, FheIntt};
///
/// // Fast polynomial multiplication
/// let ntt_a = FheNtt::new(poly_a, degree, modulus, root)?;
/// let ntt_b = FheNtt::new(poly_b, degree, modulus, root)?;
///
/// let a_ntt = ntt_a.execute().await?;
/// let b_ntt = ntt_b.execute().await?;
///
/// let pointwise_mul = FhePointwiseMul::new(a_ntt, b_ntt, degree, modulus)?;
/// let c_ntt = pointwise_mul.execute().await?;
///
/// let intt = FheIntt::new(c_ntt, degree, modulus, inv_root)?;
/// let c = intt.execute().await?;  // c = a * b (polynomial multiplication!)
/// ```
pub struct FhePointwiseMul {
    input_a: Tensor,
    input_b: Tensor,
    degree: u32,
    modulus: u64,
    barrett_mu: u64,
}

/// Parameters passed to GPU shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PointwiseMulParams {
    degree: u32,
    modulus_low: u32,
    modulus_high: u32,
    barrett_mu_low: u32,
    barrett_mu_high: u32,
    _padding: [u32; 3], // Align to 16 bytes
}

impl FhePointwiseMul {
    /// Create a new point-wise multiplication operation
    ///
    /// # Arguments
    /// * `input_a` - First polynomial in NTT domain (N × 2 u32 values)
    /// * `input_b` - Second polynomial in NTT domain (N × 2 u32 values)
    /// * `degree` - Polynomial degree (N), must be power of 2
    /// * `modulus` - FHE modulus q (64-bit prime)
    pub fn new(input_a: Tensor, input_b: Tensor, degree: u32, modulus: u64) -> Result<Self> {
        // Validate inputs
        if !degree.is_power_of_two() {
            return Err(BarracudaError::Device(format!(
                "Degree must be power of 2, got {degree}"
            )));
        }

        if !(4..=65_536).contains(&degree) {
            return Err(BarracudaError::Device(format!(
                "Degree must be in range [4, 65_536], got {degree}"
            )));
        }

        let expected_len = (degree * 2) as usize; // 2 u32 per coefficient
        if input_a.len() != expected_len {
            return Err(BarracudaError::Device(format!(
                "Input A has wrong length: expected {}, got {}",
                expected_len,
                input_a.len()
            )));
        }
        if input_b.len() != expected_len {
            return Err(BarracudaError::Device(format!(
                "Input B has wrong length: expected {}, got {}",
                expected_len,
                input_b.len()
            )));
        }

        // Ensure both tensors are on same device
        if !std::ptr::eq(input_a.device().as_ref(), input_b.device().as_ref()) {
            return Err(BarracudaError::Device(
                "input_a and input_b must be on the same device".to_string(),
            ));
        }

        // Compute Barrett reduction constant: μ = ⌊2^128 / q⌋
        // For 64-bit approximation: μ ≈ u64::MAX / q
        let barrett_mu = if modulus > 0 { u64::MAX / modulus } else { 0 };

        Ok(Self {
            input_a,
            input_b,
            degree,
            modulus,
            barrett_mu,
        })
    }

    /// Execute point-wise multiplication on GPU
    ///
    /// Returns: C = A ⊙ B (element-wise product in NTT domain)
    pub fn execute(&self) -> Result<Tensor> {
        let device = self.input_a.device();

        // Create output buffer
        let result_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Point-wise Multiply Output"),
            size: (self.degree as u64 * 2 * std::mem::size_of::<u32>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = PointwiseMulParams {
            degree: self.degree,
            modulus_low: (self.modulus & 0xFFFF_FFFF) as u32,
            modulus_high: (self.modulus >> 32) as u32,
            barrett_mu_low: (self.barrett_mu & 0xFFFF_FFFF) as u32,
            barrett_mu_high: (self.barrett_mu >> 32) as u32,
            _padding: [0; 3],
        };
        let params_buffer = device.create_uniform_buffer("FHE Pointwise Mul Params", &params);

        ComputeDispatch::new(device.as_ref(), "FHE Pointwise Mul")
            .shader(include_str!("fhe_pointwise_mul.wgsl"), "main")
            .storage_read(0, self.input_a.buffer())
            .storage_read(1, self.input_b.buffer())
            .storage_rw(2, &result_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree)
            .submit();

        // Return tensor (data stays on GPU)
        Ok(Tensor::from_buffer(
            result_buffer,
            vec![self.degree as usize * 2],
            device.clone(),
        ))
    }
}

// Tests disabled - requires integration testing framework
// Will be tested via fhe_fast_poly_mul integration tests
