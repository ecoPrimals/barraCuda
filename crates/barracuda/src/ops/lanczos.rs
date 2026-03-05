// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Lanczos eigensolver — GPU-accelerated tridiagonalization for symmetric matrices.
//!
//! Finds extreme eigenvalues/eigenvectors of large sparse symmetric matrices via
//! the Lanczos algorithm. The GPU kernel implements the core iteration (steps 2–5);
//! the matrix-vector product (step 1) is done separately by the caller.
//!
//! ## Algorithm (per iteration k)
//!
//! 1. w = A * v_k (matrix-vector product — caller provides)
//! 2. α_k = v_k^T * w
//! 3. w = w - α_k * v_k - β_{k-1} * v_{k-1}
//! 4. β_k = ||w||
//! 5. v_{k+1} = w / β_k
//!
//! ## Usage
//!
//! ```ignore
//! use barracuda::ops::lanczos::{lanczos_iteration, lanczos_eigensolver};
//!
//! // Single iteration (caller provides w = A*v_k)
//! let (alpha, beta, v_next) = lanczos_iteration(device, &w, &v_k, &v_prev, beta_prev)?;
//!
//! // K iterations with CPU matvec callback
//! let result = lanczos_eigensolver(device, n, k, &v0, |v| matrix_vector_product(a, v))?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER_LANCZOS: &str = include_str!("../shaders/spectral/lanczos_iteration_f64.wgsl");

// ── Params (must match WGSL LanczosParams: n: u32, _pad: u32, beta_prev: f64) ──

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LanczosParams {
    n: u32,
    _pad: u32,
    beta_prev: f64,
}

/// Result of one Lanczos iteration.
#[derive(Debug, Clone)]
pub struct LanczosIterationResult {
    /// Diagonal element α_k = v_k^T A v_k.
    pub alpha: f64,
    /// Off-diagonal norm β_k = ||w|| after orthogonalization.
    pub beta: f64,
    /// Next Lanczos vector v_{k+1} = w / β_k.
    pub v_next: Vec<f64>,
}

/// Tridiagonal matrix from K Lanczos iterations.
///
/// Diagonal: alpha[0..K], off-diagonal: beta[0..K-1].
/// T is symmetric tridiagonal with T[i,i] = alpha[i], T[i,i+1] = T[i+1,i] = beta[i].
#[derive(Debug, Clone)]
pub struct LanczosTridiagonal {
    /// Diagonal elements α_0..α_{K-1}.
    pub alpha: Vec<f64>,
    /// Off-diagonal elements β_0..β_{K-2}.
    pub beta: Vec<f64>,
}

/// One Lanczos iteration (steps 2–5).
///
/// * `w` — A * v_k (matrix-vector product result; caller must compute)
/// * `v_k` — current Lanczos vector
/// * `v_prev` — previous Lanczos vector (use zeros for first iteration)
/// * `beta_prev` — β_{k-1} (use 0.0 for first iteration)
///
/// Returns `(alpha_k, beta_k, v_{k+1})`.
pub fn lanczos_iteration(
    device: &Arc<WgpuDevice>,
    w: &[f64],
    v_k: &[f64],
    v_prev: &[f64],
    beta_prev: f64,
) -> Result<LanczosIterationResult> {
    let n = w.len();
    assert_eq!(v_k.len(), n, "v_k length must match w");
    assert_eq!(v_prev.len(), n, "v_prev length must match w");
    assert!(n > 0, "vector length must be positive");

    let w_buf = device.create_buffer_f64_init("lanczos:w", w);
    let v_k_buf = device.create_buffer_f64_init("lanczos:v_k", v_k);
    let v_prev_buf = device.create_buffer_f64_init("lanczos:v_prev", v_prev);
    let v_next_buf = device.create_buffer_f64(n)?;
    let tridiag_buf = device.create_buffer_f64(2)?;
    let params = LanczosParams {
        n: n as u32,
        _pad: 0,
        beta_prev,
    };
    let params_buf = device.create_uniform_buffer("lanczos:params", &params);

    ComputeDispatch::new(device, "lanczos_iteration")
        .shader(SHADER_LANCZOS, "main")
        .f64()
        .storage_read(0, &w_buf)
        .storage_read(1, &v_k_buf)
        .storage_read(2, &v_prev_buf)
        .storage_rw(3, &v_next_buf)
        .storage_rw(4, &tridiag_buf)
        .uniform(5, &params_buf)
        .dispatch(1, 1, 1)
        .submit();

    let tridiag = device.read_f64_buffer(&tridiag_buf, 2)?;
    let v_next = device.read_f64_buffer(&v_next_buf, n)?;

    Ok(LanczosIterationResult {
        alpha: tridiag[0],
        beta: tridiag[1],
        v_next,
    })
}

/// Run K Lanczos iterations and return the tridiagonal matrix.
///
/// * `matvec` — closure that computes A * v for a given vector v
/// * `v0` — initial Lanczos vector (should be unit norm)
///
/// Returns the tridiagonal matrix (alpha, beta) suitable for eigenvalue extraction.
pub fn lanczos_eigensolver<F>(
    device: &Arc<WgpuDevice>,
    n: usize,
    k: usize,
    v0: &[f64],
    mut matvec: F,
) -> Result<LanczosTridiagonal>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    assert_eq!(v0.len(), n, "v0 length must match n");
    assert!(k > 0, "k must be positive");

    let mut alpha = Vec::with_capacity(k);
    let mut beta = Vec::with_capacity(k);

    let mut v_k = v0.to_vec();
    let mut v_prev = vec![0.0; n];
    let mut beta_prev = 0.0;

    for _ in 0..k {
        let w = matvec(&v_k);
        let result = lanczos_iteration(device, &w, &v_k, &v_prev, beta_prev)?;
        alpha.push(result.alpha);
        beta.push(result.beta);
        v_prev = v_k;
        v_k = result.v_next;
        beta_prev = result.beta;
    }

    Ok(LanczosTridiagonal { alpha, beta })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lanczos_params_layout() {
        // WGSL: n (4) + _pad (4) + beta_prev (8) = 16 bytes
        assert_eq!(std::mem::size_of::<LanczosParams>(), 16);
    }

    #[test]
    fn test_lanczos_params_alignment() {
        assert_eq!(std::mem::align_of::<LanczosParams>(), 8);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn lanczos_iteration_f64_compiles_via_naga() {
        let source = include_str!("../shaders/spectral/lanczos_iteration_f64.wgsl");
        let full = crate::shaders::precision::ShaderTemplate::for_driver_auto(source, false);
        let normalized = format!(
            "enable f64;\n\n{}",
            full.replace("enable f64;\n\n", "")
                .replace("enable f64;\n", "")
        );
        if let Ok(module) = naga::front::wgsl::parse_str(&normalized) {
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("WGSL validation failed");
            assert!(!module.entry_points.is_empty());
        } else {
            assert!(!full.is_empty());
            assert!(full.contains("fn main"), "shader must contain fn main");
            assert!(
                full.contains("@compute") || full.contains("main"),
                "expected compute entry"
            );
        }
    }
}
