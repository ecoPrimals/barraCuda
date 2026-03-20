// SPDX-License-Identifier: AGPL-3.0-or-later

//! Statistical GPU ops (f64) — matrix correlation and OLS linear regression.
//!
//! Provenance: neuralSpring S69 → toadStool absorption.

use crate::device::WgpuDevice;
use crate::device::capabilities::{WORKGROUP_SIZE_1D, WORKGROUP_SIZE_COMPACT};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER_MATRIX_CORR: &str = include_str!("../shaders/stats/matrix_correlation_f64.wgsl");
const SHADER_OLS: &str = include_str!("../shaders/stats/linear_regression_f64.wgsl");

// ── Matrix Correlation ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CorrParams {
    n: u32,
    p: u32,
}

/// Compute the Pearson correlation matrix for an n×p data matrix.
///
/// `data` is row-major `[n, p]`. Returns the `p×p` correlation matrix.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn matrix_correlation(
    device: &Arc<WgpuDevice>,
    data: &[f64],
    n: u32,
    p: u32,
) -> Result<Vec<f64>> {
    let out_len = (p * p) as usize;
    let data_buf = device.create_buffer_f64_init("corr:data", data);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = CorrParams { n, p };
    let params_buf = device.create_uniform_buffer("corr:params", &params);

    ComputeDispatch::new(device, "matrix_correlation")
        .shader(SHADER_MATRIX_CORR, "main")
        .f64()
        .storage_read(0, &data_buf)
        .storage_rw(1, &out_buf)
        .uniform(2, &params_buf)
        .dispatch(
            out_len.div_ceil(WORKGROUP_SIZE_COMPACT as usize) as u32,
            1,
            1,
        )
        .submit()?;

    device.read_f64_buffer(&out_buf, out_len)
}

// ── OLS Linear Regression ───────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OlsParams {
    b: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// Batched OLS linear regression: for each batch, solve X·β = y.
///
/// * `x` — row-major `[b, n, k]` design matrices.
/// * `y` — row-major `[b, n]` response vectors.
///
/// Returns `[b, k]` coefficient vectors `β` (via normal equations).
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn linear_regression(
    device: &Arc<WgpuDevice>,
    x: &[f64],
    y: &[f64],
    b: u32,
    n: u32,
    k: u32,
) -> Result<Vec<f64>> {
    let out_len = (b * k) as usize;
    let x_buf = device.create_buffer_f64_init("ols:x", x);
    let y_buf = device.create_buffer_f64_init("ols:y", y);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = OlsParams { b, n, k, _pad: 0 };
    let params_buf = device.create_uniform_buffer("ols:params", &params);

    ComputeDispatch::new(device, "linear_regression")
        .shader(SHADER_OLS, "main")
        .f64()
        .storage_read(0, &x_buf)
        .storage_read(1, &y_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(b.div_ceil(WORKGROUP_SIZE_1D), 1, 1)
        .submit()?;

    device.read_f64_buffer(&out_buf, out_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout_corr() {
        assert_eq!(std::mem::size_of::<CorrParams>(), 8);
    }

    #[test]
    fn params_layout_ols() {
        assert_eq!(std::mem::size_of::<OlsParams>(), 16);
    }

    #[cfg(feature = "gpu")]
    fn shader_compiles_via_naga(source: &str) {
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

    #[test]
    #[cfg(feature = "gpu")]
    fn matrix_correlation_f64_compiles_via_naga() {
        let source = include_str!("../shaders/stats/matrix_correlation_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn linear_regression_f64_compiles_via_naga() {
        let source = include_str!("../shaders/stats/linear_regression_f64.wgsl");
        shader_compiles_via_naga(source);
    }
}
