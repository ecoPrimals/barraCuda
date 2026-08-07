// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neg - Pure WGSL
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

/// Neg operation
pub struct Neg {
    input: Tensor,
}

impl Neg {
    /// Create a new neg operation
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }



    /// Execute the neg operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();
        let input_buffer = self.input.buffer();
        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
        }

        let params = Params {
            size: size as u32
        };
        let params_buffer = device.create_uniform_buffer("Neg Params", &params);

        ComputeDispatch::new(device, "neg")
            .shader(
                include_str!("../shaders/math/neg_f64.wgsl"),
                "main",
            )
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
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

    /// Compute neg element-wise
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn neg_wgsl(self) -> Result<Self> {
        Neg::new(self).execute()
    }
}
