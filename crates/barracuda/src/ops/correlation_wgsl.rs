// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pearson Correlation Coefficient — GPU-Accelerated via WGSL
//!
//! Computes Pearson correlation: r = Σ(x-μx)(y-μy) / (σx·σy·n)
//!
//! **Use cases**:
//! - Feature correlation analysis (wetSpring)
//! - Sensor cross-correlation (airSpring)
//! - Observable correlation (hotSpring)
//! - Portfolio analysis
//!
//! **Note**: f32 precision. For f64, use manual computation with `weighted_dot_f64`.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for correlation shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CorrelationParams {
    size: u32,
    num_pairs: u32,
    stride: u32,
    _pad: u32,
}

/// GPU-accelerated Pearson correlation
pub struct Correlation {
    device: Arc<WgpuDevice>,
}

impl Correlation {
    /// Create a new Correlation orchestrator
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute Pearson correlation between two vectors
    /// # Arguments
    /// * `x` - First vector (f32)
    /// * `y` - Second vector (f32)
    /// # Returns
    /// Pearson correlation coefficient in [-1, 1]
    /// # Errors
    /// Returns [`Err`] if vector lengths differ, fewer than 2 elements, buffer allocation fails,
    /// GPU dispatch fails, buffer readback fails, or the device is lost.
    pub fn correlate(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        let n = x.len();
        if y.len() != n {
            return Err(BarracudaError::invalid_input(format!(
                "Vector lengths must match: x={}, y={}",
                n,
                y.len()
            )));
        }

        if n < 2 {
            return Err(BarracudaError::invalid_input(
                "Need at least 2 elements for correlation",
            ));
        }

        self.correlate_gpu(x, y)
    }

    /// Compute correlation for multiple vector pairs (batched)
    /// # Arguments
    /// * `x_batch` - Concatenated x vectors (`num_pairs` * size elements)
    /// * `y_batch` - Concatenated y vectors (`num_pairs` * size elements)
    /// * `size` - Length of each vector
    /// * `num_pairs` - Number of vector pairs
    /// # Returns
    /// Vector of correlation coefficients (one per pair)
    /// # Errors
    /// Returns [`Err`] if batch size mismatch (`x_batch/y_batch` length != `num_pairs` * size),
    /// buffer allocation fails, GPU dispatch fails, buffer readback fails, or the device is lost.
    pub fn correlate_batch(
        &self,
        x_batch: &[f32],
        y_batch: &[f32],
        size: usize,
        num_pairs: usize,
    ) -> Result<Vec<f32>> {
        if x_batch.len() != num_pairs * size || y_batch.len() != num_pairs * size {
            return Err(BarracudaError::invalid_input(format!(
                "Batch size mismatch: expected {} elements, got x={}, y={}",
                num_pairs * size,
                x_batch.len(),
                y_batch.len()
            )));
        }

        self.correlate_batch_gpu(x_batch, y_batch, size, num_pairs)
    }

    /// CPU reference implementation
    #[expect(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn correlate_cpu(&self, x: &[f32], y: &[f32]) -> f32 {
        let n = x.len() as f32;
        let mean_x: f32 = x.iter().sum::<f32>() / n;
        let mean_y: f32 = y.iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_x = 0.0f32;
        let mut var_y = 0.0f32;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov = dx.mul_add(dy, cov);
            var_x = dx.mul_add(dx, var_x);
            var_y = dy.mul_add(dy, var_y);
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-10_f32 {
            return 0.0;
        }
        cov / denom
    }

    fn correlate_gpu(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        let results = self.correlate_batch_gpu(x, y, x.len(), 1)?;
        Ok(results[0])
    }

    fn correlate_batch_gpu(
        &self,
        x_batch: &[f32],
        y_batch: &[f32],
        size: usize,
        num_pairs: usize,
    ) -> Result<Vec<f32>> {
        // Create buffers
        let x_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("X"),
                contents: bytemuck::cast_slice(x_batch),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let y_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Y"),
                contents: bytemuck::cast_slice(y_batch),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output"),
            size: (num_pairs * 4) as u64, // f32 = 4 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = CorrelationParams {
            size: size as u32,
            num_pairs: num_pairs as u32,
            stride: size as u32,
            _pad: 0,
        };

        let params_buf = self.device.create_uniform_buffer("Params", &params);

        ComputeDispatch::new(&self.device, "Correlation")
            .shader(include_str!("../shaders/special/correlation.wgsl"), "main")
            .storage_read(0, &x_buf)
            .storage_read(1, &y_buf)
            .storage_rw(2, &output_buf)
            .uniform(3, &params_buf)
            .dispatch_1d(num_pairs as u32)
            .submit()?;

        // Read back results
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (num_pairs * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder2 = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });
        encoder2.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, (num_pairs * 4) as u64);
        self.device.submit_commands(Some(encoder2.finish()));

        let results: Vec<f32> = self.device.map_staging_buffer(&staging, num_pairs)?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_perfect_positive() {
        let device = crate::device::test_pool::get_test_device_sync();
        let op = Correlation::new(device).unwrap();

        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..100).map(|i| (i as f32).mul_add(2.0, 1.0)).collect();

        let r = op.correlate(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 0.001, "Expected r≈1.0, got {r}");
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let device = crate::device::test_pool::get_test_device_sync();
        let op = Correlation::new(device).unwrap();

        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..100).map(|i| -(i as f32)).collect();

        let r = op.correlate(&x, &y).unwrap();
        assert!((r + 1.0).abs() < 0.001, "Expected r≈-1.0, got {r}");
    }

    #[test]
    fn test_correlation_uncorrelated() {
        let device = crate::device::test_pool::get_test_device_sync();
        let op = Correlation::new(device).unwrap();

        // Sin and cos are orthogonal
        let n = 1000;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();

        let r = op.correlate(&x, &y).unwrap();
        assert!(r.abs() < 0.1, "Expected r≈0, got {r}");
    }
}
