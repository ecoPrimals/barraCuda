// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pow - Element-wise power operation - Pure WGSL
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

/// Power operation
pub struct Pow {
    input: Tensor,
    exponent: f32,
}

impl Pow {
    /// Create a new pow operation
    #[must_use]
    pub fn new(input: Tensor, exponent: f32) -> Self {
        Self { input, exponent }
    }

    /// Execute the pow operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            exponent: f32,
        }

        let params = Params {
            size: size as u32,
            exponent: self.exponent,
        };

        let params_buffer = device.create_uniform_buffer("Pow Params", &params);

        ComputeDispatch::new(device, "Pow")
            .shader(include_str!("../shaders/math/pow_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute element-wise power
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn pow_wgsl(self, exponent: f32) -> Result<Self> {
        Pow::new(self, exponent).execute()
    }
}
