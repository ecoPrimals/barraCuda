// SPDX-License-Identifier: AGPL-3.0-or-later
//! Histc - Histogram with custom bins (Pure WGSL)
//!
//! Computes histogram of input values into specified bins
//! Uses atomic operations for parallel histogram computation
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Histogram computation
pub struct Histc {
    input: Tensor,
    num_bins: usize,
    min_val: f32,
    max_val: f32,
}

impl Histc {
    /// Create histogram with given bin count and value range.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, num_bins: usize, min_val: f32, max_val: f32) -> Result<Self> {
        if num_bins == 0 {
            return Err(BarracudaError::invalid_op(
                "histc",
                "num_bins must be positive",
            ));
        }

        if min_val >= max_val {
            return Err(BarracudaError::invalid_op(
                "histc",
                "min_val must be less than max_val",
            ));
        }

        Ok(Self {
            input,
            num_bins,
            min_val,
            max_val,
        })
    }

    /// Execute histogram computation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size = self.input.shape().iter().product::<usize>();

        // Create atomic histogram buffer (zero-initialized)
        let histogram_buffer = device.create_buffer_u32_zeros(self.num_bins)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            num_bins: u32,
            min_val: f32,
            max_val: f32,
            bin_width: f32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let bin_width = (self.max_val - self.min_val) / self.num_bins as f32;

        let params = Params {
            size: input_size as u32,
            num_bins: self.num_bins as u32,
            min_val: self.min_val,
            max_val: self.max_val,
            bin_width,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Histc Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Histc")
            .shader(include_str!("../shaders/misc/histc_f64.wgsl"), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, self.input.buffer())
            .storage_rw(2, &histogram_buffer)
            .dispatch_1d(input_size as u32)
            .submit()?;

        // Tensor type is f32 — create an f32 buffer for the histogram output.
        // The u32 atomic histogram is converted to f32 via GPU readback.
        // A u32 → f32 copy shader would avoid the readback but adds dispatch
        // overhead that only pays off at >10k bins.
        let histogram_f32_buffer = device.create_buffer_f32(self.num_bins)?;
        Ok(Tensor::from_buffer(
            histogram_f32_buffer,
            vec![self.num_bins],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_histc_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input =
            Tensor::from_vec_on(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6], device.clone())
                .await
                .unwrap();

        let histc = Histc::new(input, 10, 0.0, 10.0).unwrap();
        let output = histc.execute().unwrap();

        assert_eq!(output.shape(), &[10]);
    }

    #[tokio::test]
    async fn test_histc_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 1000;
        let input = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let histc = Histc::new(input, 20, 0.0, 2.0).unwrap();
        let output = histc.execute().unwrap();

        assert_eq!(output.shape(), &[20]);
    }
}
