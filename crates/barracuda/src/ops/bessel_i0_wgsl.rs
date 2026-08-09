// SPDX-License-Identifier: AGPL-3.0-or-later
//! BESSEL I0 - Modified Bessel function of the first kind, order 0
//!
//! I₀(x) for cylindrical coordinate physics. Uses polynomial approximation
//! from Abramowitz & Stegun 9.8.1-9.8.2.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Modified Bessel function of the first kind, order 0: I₀(x).
pub struct BesselI0 {
    input: Tensor,
}

impl BesselI0 {
    /// Create a new besseli0 operation
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    /// Execute Bessel I0 on GPU.
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
        let metadata_buffer = device.create_uniform_buffer("BESSEL_I0 Metadata", &metadata);

        ComputeDispatch::new(device, "besseli0")
            .shader(include_str!("../shaders/special/bessel_i0.wgsl"), "main")
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
    /// Compute element-wise I₀(x). I₀(0) = 1.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn bessel_i0(self) -> Result<Self> {
        BesselI0::new(self).execute()
    }
}
