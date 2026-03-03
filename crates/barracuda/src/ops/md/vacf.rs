// SPDX-License-Identifier: AGPL-3.0-or-later

//! VACF batch GPU op — Velocity Autocorrelation Function via ComputeDispatch.
//!
//! Computes C(τ) = (1/n_origins) Σₜ v(t)·v(t+τ) for each lag τ.
//! Each thread handles one lag value.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER_VACF_BATCH: &str = include_str!("../../shaders/md/vacf_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VacfBatchParams {
    n_atoms: u32,
    n_frames: u32,
    n_lags: u32,
    _pad: u32,
}

/// Compute VACF C(τ) for a batch of velocity snapshots.
///
/// Returns `C(τ)` for τ = 0..n_lags-1.
///
/// # Arguments
/// * `velocities` — `[n_frames × N × 3]` f64, flattened (frame-major)
/// * `n_atoms` — number of particles
/// * `n_frames` — number of velocity snapshots
/// * `n_lags` — number of lag values to compute
pub fn compute_vacf_batch(
    device: &Arc<WgpuDevice>,
    velocities: &[f64],
    n_atoms: usize,
    n_frames: usize,
    n_lags: usize,
) -> Result<Vec<f64>> {
    let expected_len = n_frames * n_atoms * 3;
    assert_eq!(
        velocities.len(),
        expected_len,
        "velocities must be n_frames × n_atoms × 3"
    );
    assert!(n_lags <= n_frames, "n_lags must be <= n_frames");

    let vel_buf = device.create_buffer_f64_init("vacf_batch:vel", velocities);
    let out_buf = device.create_buffer_f64(n_lags)?;

    let params = VacfBatchParams {
        n_atoms: n_atoms as u32,
        n_frames: n_frames as u32,
        n_lags: n_lags as u32,
        _pad: 0,
    };
    let params_buf = device.create_uniform_buffer("vacf_batch:params", &params);

    ComputeDispatch::new(device, "vacf_batch")
        .shader(SHADER_VACF_BATCH, "main")
        .f64()
        .storage_read(0, &vel_buf)
        .storage_rw(1, &out_buf)
        .uniform(2, &params_buf)
        .dispatch_1d(n_lags as u32)
        .submit();

    device.read_f64_buffer(&out_buf, n_lags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacf_params_layout() {
        // repr(C): n_atoms(u32) + n_frames(u32) + n_lags(u32) + _pad(u32) = 16 bytes
        let params = VacfBatchParams {
            n_atoms: 100,
            n_frames: 500,
            n_lags: 100,
            _pad: 0,
        };
        assert_eq!(std::mem::size_of::<VacfBatchParams>(), 16);
        assert_eq!(std::mem::align_of::<VacfBatchParams>(), 4);
        assert_eq!(params.n_atoms, 100);
    }

    #[test]
    fn test_vacf_shader_valid() {
        assert!(!SHADER_VACF_BATCH.is_empty(), "shader must not be empty");
        assert!(
            SHADER_VACF_BATCH.contains("fn main"),
            "shader must contain 'fn main'"
        );
    }
}
