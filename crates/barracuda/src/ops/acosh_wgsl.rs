// SPDX-License-Identifier: AGPL-3.0-or-later
//! ACOSH - Inverse hyperbolic cosine operation - Pure WGSL
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

/// Element-wise inverse hyperbolic cosine: acosh(x), valid for x ≥ 1.
pub struct Acosh {
    input: Tensor,
}

impl Acosh {
    /// Create a new acosh operation
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }



    /// Execute acosh on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();
        let input_buffer = self.input.buffer();
        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Metadata {
            size: u32,
        }

        let metadata = Metadata {
            size: size as u32
        };
        let metadata_buffer = device.create_uniform_buffer("ACOSH Metadata", &metadata);

        ComputeDispatch::new(device, "acosh")
            .shader(
                include_str!("../shaders/math/acosh_f64.wgsl"),
                "main",
            )
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &metadata_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {

    /// Compute element-wise acosh(x).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn acosh(self) -> Result<Self> {
        Acosh::new(self).execute()
    }
}
