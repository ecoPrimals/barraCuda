// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-accelerated batched Brent root-finding at f64 precision.
//!
//! Solves many independent root-finding problems in parallel on GPU using
//! Brent's method — combining bisection, secant, and inverse quadratic
//! interpolation for superlinear convergence with guaranteed reliability.
//!
//! Built-in residual functions:
//! - VG inverse: van Genuchten θ(h) = target  (soil hydrology)
//! - Green-Ampt: cumulative infiltration F(t) = target
//! - Polynomial: x² = target  (validation / custom)
//!
//! Provenance: airSpring V035/V045 handoff → toadStool absorption

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::utils::chunk_to_array;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Built-in residual function selector for batched Brent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrentFunction {
    /// Van Genuchten inverse: θ(h) - target = 0.
    /// Aux params: (`θ_r`, `θ_s`, α, n).
    VanGenuchtenInverse,
    /// Green-Ampt ponding time: F(t) - target = 0.
    /// Aux params: (`K_s`, `ψ_f·Δθ`, _, _).
    GreenAmpt,
    /// Polynomial: x² - target = 0 (validation).
    Polynomial,
}

impl BrentFunction {
    fn operation_id(self) -> u32 {
        match self {
            Self::VanGenuchtenInverse => 0,
            Self::GreenAmpt => 1,
            Self::Polynomial => 2,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BrentParams {
    batch_size: u32,
    max_iter: u32,
    operation: u32,
    _pad: u32,
    tol: f64,
    aux_a: f64,
    aux_b: f64,
    aux_c: f64,
    aux_d: f64,
}

/// GPU-accelerated batched Brent root-finding.
///
/// Each problem finds x in [a, b] such that f(x) = target, using Brent's
/// method for superlinear convergence with guaranteed reliability.
pub struct BrentGpu {
    device: Arc<WgpuDevice>,
    max_iterations: u32,
    tolerance: f64,
}

/// Result of batched Brent root-finding.
pub struct BrentGpuResult {
    /// Found roots [`batch_size`]
    pub roots: Vec<f64>,
    /// Iterations used per problem [`batch_size`]
    pub iterations: Vec<u32>,
}

impl BrentGpu {
    /// Create a new batched Brent solver.
    /// # Arguments
    /// * `device` - GPU device
    /// * `max_iterations` - Maximum iterations per problem (typically 50-100)
    /// * `tolerance` - Convergence tolerance (typically 1e-10 to 1e-14)
    /// # Errors
    /// Returns [`Err`] if `tolerance` ≤ 0.
    pub fn new(device: Arc<WgpuDevice>, max_iterations: u32, tolerance: f64) -> Result<Self> {
        if tolerance <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "Tolerance must be positive".to_string(),
            });
        }
        Ok(Self {
            device,
            max_iterations,
            tolerance,
        })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/optimize/brent_f64.wgsl")
    }

    /// Solve polynomial roots: x² = target for each problem.
    /// # Errors
    /// Returns [`Err`] if `lower`, `upper`, and `targets` lengths do not match, or if GPU
    /// execution or buffer readback fails (e.g., device lost).
    pub fn solve_polynomial(
        &self,
        lower: &[f64],
        upper: &[f64],
        targets: &[f64],
    ) -> Result<BrentGpuResult> {
        self.solve(
            lower,
            upper,
            targets,
            BrentFunction::Polynomial,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Solve van Genuchten inverse: find h where θ(h) = target.
    /// # Errors
    /// Returns [`Err`] if array lengths do not match, or if GPU execution or buffer readback fails.
    pub fn solve_vg_inverse(
        &self,
        lower: &[f64],
        upper: &[f64],
        targets: &[f64],
        theta_r: f64,
        theta_s: f64,
        alpha: f64,
        n: f64,
    ) -> Result<BrentGpuResult> {
        self.solve(
            lower,
            upper,
            targets,
            BrentFunction::VanGenuchtenInverse,
            theta_r,
            theta_s,
            alpha,
            n,
        )
    }

    /// Solve Green-Ampt: find F where cumulative infiltration equation holds.
    /// # Errors
    /// Returns [`Err`] if array lengths do not match, or if GPU execution or buffer readback fails.
    pub fn solve_green_ampt(
        &self,
        lower: &[f64],
        upper: &[f64],
        targets: &[f64],
        k_s: f64,
        psi_delta_theta: f64,
    ) -> Result<BrentGpuResult> {
        self.solve(
            lower,
            upper,
            targets,
            BrentFunction::GreenAmpt,
            k_s,
            psi_delta_theta,
            0.0,
            0.0,
        )
    }

    fn solve(
        &self,
        lower: &[f64],
        upper: &[f64],
        targets: &[f64],
        func: BrentFunction,
        aux_a: f64,
        aux_b: f64,
        aux_c: f64,
        aux_d: f64,
    ) -> Result<BrentGpuResult> {
        let batch_size = lower.len();
        if upper.len() != batch_size || targets.len() != batch_size {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Array lengths must match: lower={}, upper={}, targets={}",
                    lower.len(),
                    upper.len(),
                    targets.len()
                ),
            });
        }
        if batch_size == 0 {
            return Ok(BrentGpuResult {
                roots: vec![],
                iterations: vec![],
            });
        }

        let lower_bytes: &[u8] = bytemuck::cast_slice(lower);
        let lower_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Brent lower"),
                    contents: lower_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let upper_bytes: &[u8] = bytemuck::cast_slice(upper);
        let upper_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Brent upper"),
                    contents: upper_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let target_bytes: &[u8] = bytemuck::cast_slice(targets);
        let target_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Brent target"),
                    contents: target_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let roots_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brent roots"),
            size: (batch_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let iterations_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brent iterations"),
            size: (batch_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = BrentParams {
            batch_size: batch_size as u32,
            max_iter: self.max_iterations,
            operation: func.operation_id(),
            _pad: 0,
            tol: self.tolerance,
            aux_a,
            aux_b,
            aux_c,
            aux_d,
        };
        let params_buffer = self.device.create_uniform_buffer("Brent params", &params);

        ComputeDispatch::new(self.device.as_ref(), "brent_solve")
            .shader(Self::wgsl_shader(), "brent_solve")
            .f64()
            .storage_read(0, &lower_buffer)
            .storage_read(1, &upper_buffer)
            .storage_read(2, &target_buffer)
            .storage_rw(3, &roots_buffer)
            .storage_rw(4, &iterations_buffer)
            .uniform(5, &params_buffer)
            .dispatch(batch_size as u32, 1, 1)
            .submit()?;

        let roots = self.device.read_f64_buffer(&roots_buffer, batch_size)?;
        let iterations = self.read_u32_buffer(&iterations_buffer, batch_size)?;

        Ok(BrentGpuResult { roots, iterations })
    }

    fn read_u32_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brent u32 staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Brent u32 readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
        self.device.submit_commands(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll_safe()?;
        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("GPU buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<u32> = data
            .chunks_exact(4)
            .map(|chunk| chunk_to_array::<4>(chunk).map(u32::from_le_bytes))
            .collect::<Result<Vec<_>>>()?;
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_gpu_device;

    #[tokio::test]
    async fn test_brent_gpu_sqrt() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let brent = BrentGpu::new(device, 100, 1e-12).unwrap();

        let lower = vec![0.0, 0.0, 0.0, 1.0];
        let upper = vec![2.0, 2.0, 3.0, 4.0];
        let targets = vec![2.0, 3.0, 5.0, 9.0];

        let result = brent.solve_polynomial(&lower, &upper, &targets).unwrap();

        let expected = [2.0_f64.sqrt(), 3.0_f64.sqrt(), 5.0_f64.sqrt(), 3.0];
        for (i, (&root, &exp)) in result.roots.iter().zip(expected.iter()).enumerate() {
            assert!(
                (root - exp).abs() < 1e-8,
                "root[{i}] = {root}, expected {exp}"
            );
        }
    }

    #[tokio::test]
    async fn test_brent_gpu_empty() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let brent = BrentGpu::new(device, 100, 1e-12).unwrap();
        let result = brent.solve_polynomial(&[], &[], &[]).unwrap();
        assert!(result.roots.is_empty());
    }
}
