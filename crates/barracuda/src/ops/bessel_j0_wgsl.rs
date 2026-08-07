// SPDX-License-Identifier: AGPL-3.0-or-later
//! BESSEL J0 - Bessel function of the first kind, order 0
//!
//! J₀(x) for cylindrical coordinate physics. Uses rational polynomial
//! approximation from Abramowitz & Stegun 9.4.1-9.4.3.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Bessel function of the first kind, order 0: J₀(x).
pub struct BesselJ0 {
    input: Tensor,
}

impl BesselJ0 {
    /// Create a new besselj0 operation
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }



    /// Execute Bessel J0 on GPU.
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
        struct Metadata {
            size: u32,
        }

        let metadata = Metadata {
            size: size as u32
        };
        let metadata_buffer = device.create_uniform_buffer("BESSEL_J0 Metadata", &metadata);

        ComputeDispatch::new(device, "besselj0")
            .shader(
                include_str!("../shaders/special/bessel_j0.wgsl"),
                "main",
            )
            .storage_read(0, input_buffer)
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

    /// Compute element-wise J₀(x). J₀(0) = 1.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn bessel_j0(self) -> Result<Self> {
        BesselJ0::new(self).execute()
    }
}
