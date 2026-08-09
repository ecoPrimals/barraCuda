// SPDX-License-Identifier: AGPL-3.0-or-later

//! FOCE (First-Order Conditional Estimation) per-subject gradient — GPU kernel.
//!
//! Computes the per-subject gradient of the FOCE objective function for
//! population PK estimation. Each subject is independent, making this
//! embarrassingly parallel on the GPU.
//!
//! At 1,000+ subjects, GPU execution is 50–100× faster than sequential CPU.
//!
//! Provenance: healthSpring V14 → barraCuda absorption (Mar 2026)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;

/// WGSL shader for FOCE per-subject gradient computation.
pub const WGSL_FOCE_GRADIENT: &str = include_str!("../../shaders/pharma/foce_gradient_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FoceConfig {
    n_subjects: u32,
    n_obs_max: u32,
    n_params: u32,
    _pad: u32,
}

/// Result of a FOCE gradient computation.
#[derive(Debug, Clone)]
pub struct FoceGradientResult {
    /// Per-subject objective function values `[n_subjects]`.
    pub objectives: Vec<f64>,
    /// Per-subject gradients `[n_subjects × n_params]`.
    pub gradients: Vec<f64>,
}

/// GPU kernel for FOCE per-subject gradient computation.
pub struct FoceGradientGpu {
    device: Arc<WgpuDevice>,
}

impl FoceGradientGpu {
    /// Create the FOCE gradient kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute FOCE per-subject gradients.
    ///
    /// # Arguments
    /// - `residuals`: `[n_subjects × n_obs_max]` — prediction residuals
    /// - `variances`: `[n_subjects × n_obs_max]` — residual variances
    /// - `jacobian`: `[n_subjects × n_obs_max × n_params]` — Jacobian matrix
    /// - `obs_counts`: `[n_subjects]` — number of observations per subject
    /// - `n_subjects`, `n_obs_max`, `n_params`: dimensions
    ///
    /// # Errors
    /// Returns [`Err`] if the device is lost or poll fails.
    pub fn compute(
        &self,
        residuals: &[f64],
        variances: &[f64],
        jacobian: &[f64],
        obs_counts: &[u32],
        n_subjects: u32,
        n_obs_max: u32,
        n_params: u32,
    ) -> Result<FoceGradientResult> {
        let config = FoceConfig {
            n_subjects,
            n_obs_max,
            n_params,
            _pad: 0,
        };

        let d = self.device.device();

        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foce_config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let residuals_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foce_residuals"),
            contents: bytemuck::cast_slice(residuals),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let variances_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foce_variances"),
            contents: bytemuck::cast_slice(variances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let jacobian_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foce_jacobian"),
            contents: bytemuck::cast_slice(jacobian),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let obs_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foce_obs_counts"),
            contents: bytemuck::cast_slice(obs_counts),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let grad_size = (n_subjects * n_params) as u64 * 8;
        let obj_size = n_subjects as u64 * 8;

        let grad_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foce_gradients"),
            size: grad_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let obj_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foce_objectives"),
            size: obj_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ComputeDispatch::new(&self.device, "FoceGradient")
            .shader(WGSL_FOCE_GRADIENT, "main")
            .f64()
            .uniform(0, &config_buf)
            .storage_read(1, &residuals_buf)
            .storage_read(2, &variances_buf)
            .storage_read(3, &jacobian_buf)
            .storage_read(4, &obs_buf)
            .storage_rw(5, &grad_buf)
            .storage_rw(6, &obj_buf)
            .dispatch_1d(n_subjects)
            .submit()?;

        let grad_staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foce_grad_staging"),
            size: grad_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let obj_staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foce_obj_staging"),
            size: obj_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("foce_readback"),
            });
        encoder.copy_buffer_to_buffer(&grad_buf, 0, &grad_staging, 0, grad_size);
        encoder.copy_buffer_to_buffer(&obj_buf, 0, &obj_staging, 0, obj_size);
        self.device
            .queue()
            .submit(std::iter::once(encoder.finish()));
        self.device.poll_safe()?;

        let gradients = self.read_f64_buffer(&grad_staging, (n_subjects * n_params) as usize)?;
        let objectives = self.read_f64_buffer(&obj_staging, n_subjects as usize)?;

        Ok(FoceGradientResult {
            objectives,
            gradients,
        })
    }

    fn read_f64_buffer(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<f64>> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll_safe()?;
        rx.recv()
            .map_err(|_| crate::error::BarracudaError::device_lost("readback channel closed"))?
            .map_err(|e| {
                crate::error::BarracudaError::DeviceLost(format!("buffer map failed: {e:?}"))
            })?;
        let data = slice.get_mapped_range();
        let values: Vec<f64> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        buf.unmap();
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_layout() {
        assert_eq!(std::mem::size_of::<FoceConfig>(), 16);
    }
}
