// SPDX-License-Identifier: AGPL-3.0-only
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

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
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
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/correlation.wgsl")
    }

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
            return Err(BarracudaError::InvalidInput {
                message: format!("Vector lengths must match: x={}, y={}", n, y.len()),
            });
        }

        if n < 2 {
            return Err(BarracudaError::InvalidInput {
                message: "Need at least 2 elements for correlation".to_string(),
            });
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
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Batch size mismatch: expected {} elements, got x={}, y={}",
                    num_pairs * size,
                    x_batch.len(),
                    y_batch.len()
                ),
            });
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
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
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
        let shader = self
            .device
            .compile_shader(Self::wgsl_shader(), Some("Correlation"));

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

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout
        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Correlation BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pl = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Correlation PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Correlation Pipeline"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Correlation BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        // Dispatch
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Correlation Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Correlation Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            let n_workgroups = num_pairs.div_ceil(WORKGROUP_SIZE_1D as usize);
            pass.dispatch_workgroups(n_workgroups as u32, 1, 1);
        }

        self.device.submit_and_poll(Some(encoder.finish()));

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
        self.device.submit_and_poll(Some(encoder2.finish()));

        let results: Vec<f32> = self.device.map_staging_buffer(&staging, num_pairs)?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Arc<crate::device::WgpuDevice> {
        crate::device::test_pool::get_test_device_sync()
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let device = get_test_device();
        let op = Correlation::new(device).unwrap();

        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..100).map(|i| i as f32 * 2.0 + 1.0).collect();

        let r = op.correlate(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 0.001, "Expected r≈1.0, got {r}");
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let device = get_test_device();
        let op = Correlation::new(device).unwrap();

        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..100).map(|i| -(i as f32)).collect();

        let r = op.correlate(&x, &y).unwrap();
        assert!((r + 1.0).abs() < 0.001, "Expected r≈-1.0, got {r}");
    }

    #[test]
    fn test_correlation_uncorrelated() {
        let device = get_test_device();
        let op = Correlation::new(device).unwrap();

        // Sin and cos are orthogonal
        let n = 1000;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();

        let r = op.correlate(&x, &y).unwrap();
        assert!(r.abs() < 0.1, "Expected r≈0, got {r}");
    }
}
