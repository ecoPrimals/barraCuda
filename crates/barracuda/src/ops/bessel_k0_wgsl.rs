// SPDX-License-Identifier: AGPL-3.0-or-later
//! BESSEL K0 - Modified Bessel function of the third kind, order 0
//!
//! K₀(x) for cylindrical coordinate physics. Uses polynomial approximation
//! from Abramowitz & Stegun 9.8.3-9.8.6. Returns infinity for x <= 0.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Modified Bessel function of the third kind, order 0: K₀(x).
pub struct BesselK0 {
    input: Tensor,
}

impl BesselK0 {
    /// Create a new besselk0 operation
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    /// Execute Bessel K0 on GPU.
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

        let metadata = Metadata { size: size as u32 };
        let metadata_buffer = device.create_uniform_buffer("BESSEL_K0 Metadata", &metadata);

        ComputeDispatch::new(device, "besselk0")
            .shader(include_str!("../shaders/special/bessel_k0.wgsl"), "main")
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
    /// Compute element-wise K₀(x). Returns infinity for x ≤ 0.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn bessel_k0(self) -> Result<Self> {
        BesselK0::new(self).execute()
    }
}
