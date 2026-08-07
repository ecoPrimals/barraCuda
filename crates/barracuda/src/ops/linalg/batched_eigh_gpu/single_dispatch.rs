// SPDX-License-Identifier: AGPL-3.0-or-later
//! Single-dispatch Jacobi eigensolver — one GPU submit for n≤32
//!
//! Eliminates poll bottleneck: instead of ~8000 `queue.submit()` calls per batch,
//! uses exactly ONE dispatch. Limited to n≤32 by workgroup shared memory.

use super::BatchedEighGpu;
use super::params::SingleDispatchParams;
use crate::device::WgpuDevice;
use crate::device::capabilities::{DeviceCapabilities, EigensolveStrategy};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::shaders::precision::ShaderTemplate;
use std::sync::Arc;

impl BatchedEighGpu {
    /// **SINGLE-DISPATCH** batched eigenvalue decomposition — eliminates poll bottleneck
    /// For n=12, batch=40: Previous 7,920 submits → This: 1 submit.
    /// Maximum n=32 (workgroup shared memory limit).
    /// # Errors
    /// Returns [`Err`] if `n > 32`, `data.len() != batch_size * n * n`, buffer allocation fails, pipeline execution fails, or the device is lost.
    pub fn execute_single_dispatch(
        device: Arc<WgpuDevice>,
        data: &[f64],
        n: usize,
        batch_size: usize,
        max_sweeps: u32,
        tolerance: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        const MAX_N: usize = 32;

        if n > MAX_N {
            return Err(BarracudaError::invalid_input(format!(
                "Single-dispatch eigensolve limited to n≤{MAX_N}, got n={n}. Use execute_f64() for larger matrices."
            )));
        }

        if data.len() != batch_size * n * n {
            return Err(BarracudaError::invalid_input(format!(
                "Data length {} does not match batch_size={} × n²={}",
                data.len(),
                batch_size,
                n * n
            )));
        }

        let a_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SingleDispatch A"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let v_size = (batch_size * n * n * 8) as u64;
        let v_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SingleDispatch V"),
            size: v_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let eig_size = (batch_size * n * 8) as u64;
        let eig_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SingleDispatch Eigenvalues"),
            size: eig_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = SingleDispatchParams {
            n: n as u32,
            batch_size: batch_size as u32,
            max_sweeps,
            tolerance: tolerance as f32,
        };
        let params_buffer = device.create_uniform_buffer("SingleDispatch Params", &params);

        let caps = DeviceCapabilities::from_device(&device);
        let wave_size = match caps.optimal_eigensolve_strategy() {
            EigensolveStrategy::WarpPacked { wg_size } => wg_size,
            EigensolveStrategy::WavePacked { wave_size } => wave_size,
            EigensolveStrategy::Standard => 1,
        };
        let patched_shader = ShaderTemplate::patch_warp_size(
            Self::single_dispatch_shader_for_device(&device),
            wave_size,
        );

        ComputeDispatch::new(&device, "SingleDispatch")
            .shader(&patched_shader, "batched_eigh_single_dispatch")
            .f64()
            .uniform(0, &params_buffer)
            .storage_rw(1, &a_buffer)
            .storage_rw(2, &v_buffer)
            .storage_rw(3, &eig_buffer)
            .dispatch((batch_size as u32).div_ceil(wave_size), 1, 1)
            .submit()?;

        let eigenvalues = device.read_f64_buffer(&eig_buffer, batch_size * n)?;
        let eigenvectors = device.read_f64_buffer(&v_buffer, batch_size * n * n)?;

        Ok((eigenvalues, eigenvectors))
    }

    /// **SINGLE-DISPATCH** buffer-based eigensolve — no CPU readback.
    /// Execute batched eigenvalue decomposition using pre-allocated buffers.
    /// # Errors
    /// Returns [`Err`] if `n > 32`, buffer sizes are invalid, shader compilation fails, pipeline execution fails, or the device is lost.
    pub fn execute_single_dispatch_buffers(
        device: &Arc<WgpuDevice>,
        matrices_buffer: &wgpu::Buffer,
        eigenvalues_buffer: &wgpu::Buffer,
        eigenvectors_buffer: &wgpu::Buffer,
        n: usize,
        batch_size: usize,
        max_sweeps: u32,
        tolerance: f64,
    ) -> Result<()> {
        const MAX_N: usize = 32;

        if n > MAX_N {
            return Err(BarracudaError::invalid_input(format!(
                "Single-dispatch eigensolve limited to n≤{MAX_N}, got n={n}"
            )));
        }

        let params = SingleDispatchParams {
            n: n as u32,
            batch_size: batch_size as u32,
            max_sweeps,
            tolerance: tolerance as f32,
        };
        let params_buffer =
            device.create_uniform_buffer("SingleDispatch Params (buffers)", &params);

        let caps = DeviceCapabilities::from_device(device);
        let wave_size = match caps.optimal_eigensolve_strategy() {
            EigensolveStrategy::WarpPacked { wg_size } => wg_size,
            EigensolveStrategy::WavePacked { wave_size } => wave_size,
            EigensolveStrategy::Standard => 1,
        };
        let patched_shader = ShaderTemplate::patch_warp_size(
            Self::single_dispatch_shader_for_device(device),
            wave_size,
        );

        ComputeDispatch::new(device, "SingleDispatch")
            .shader(&patched_shader, "batched_eigh_single_dispatch")
            .f64()
            .uniform(0, &params_buffer)
            .storage_rw(1, matrices_buffer)
            .storage_rw(2, eigenvectors_buffer)
            .storage_rw(3, eigenvalues_buffer)
            .dispatch((batch_size as u32).div_ceil(wave_size), 1, 1)
            .submit()?;

        Ok(())
    }
}
