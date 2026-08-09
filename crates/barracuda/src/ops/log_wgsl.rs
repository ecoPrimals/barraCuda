// SPDX-License-Identifier: AGPL-3.0-or-later
//! Log — Pure WGSL via ComputeDispatch builder.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Log operation
pub struct Log {
    input: Tensor,
}

impl Log {
    /// Create a new log operation
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    /// Execute the log operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();
        let input_buffer = self.input.buffer();
        let output_buffer = device.create_buffer_f32(size)?;

        let params = [size as u32];
        let params_buffer = device.create_uniform_buffer("Log Params", &params);

        ComputeDispatch::new(device, "log")
            .shader(include_str!("../shaders/math/log_f64.wgsl"), "main")
            .storage_read(0, input_buffer)
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
    /// Compute log element-wise
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn log_wgsl(self) -> Result<Self> {
        Log::new(self).execute()
    }
}
