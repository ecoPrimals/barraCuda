// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conjugate Gradient vector operations for lattice QCD.
//!
//! Three BLAS-like GPU kernels for the CG solver on complex fermion fields:
//!
//! | Kernel | Operation | Use in CG |
//! |--------|-----------|-----------|
//! | `complex_dot_re` | `out[i] = Re(a[i]* × b[i])` | Inner products `<r|r>`, `<p|Ap>` |
//! | `axpy` | `y[i] += α × x[i]` | Solution update `x += α·p` |
//! | `xpay` | `p[i] = x[i] + β × p[i]` | Direction update `p = r + β·p` |
//!
//! ## Absorbed from
//!
//! hotSpring v0.6.1 `lattice/cg.rs` (Feb 2026) — 9/9 CG suite PASS,
//! convergence on cold and hot SU(3) lattices.

/// WGSL source for the three CG kernels (separate entry points).
pub const WGSL_CG_KERNELS_F64: &str = include_str!("../../shaders/lattice/cg_kernels_f64.wgsl");

// GPU-resident CG shaders (alpha/beta on GPU, no per-iteration readback).
// Absorbed from hotSpring lattice QCD (Feb 2026).
pub const WGSL_SUM_REDUCE_F64: &str = include_str!("../../shaders/lattice/sum_reduce_f64.wgsl");
pub const WGSL_CG_COMPUTE_ALPHA_F64: &str =
    include_str!("../../shaders/lattice/cg_compute_alpha_f64.wgsl");
pub const WGSL_CG_COMPUTE_BETA_F64: &str =
    include_str!("../../shaders/lattice/cg_compute_beta_f64.wgsl");
pub const WGSL_CG_UPDATE_XR_F64: &str = include_str!("../../shaders/lattice/cg_update_xr_f64.wgsl");
pub const WGSL_CG_UPDATE_P_F64: &str = include_str!("../../shaders/lattice/cg_update_p_f64.wgsl");

/// WGSL fragment: real part of complex dot product.
///
/// `out[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]`
///
/// Sum `out[0..n_pairs]` via `ReduceScalarPipeline` to obtain `Re(<a|b>)`.
pub const WGSL_COMPLEX_DOT_RE_F64: &str = r"
struct Params { n_pairs: u32, pad0: u32, pad1: u32, pad2: u32, }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f64>;
@group(0) @binding(2) var<storage, read> b: array<f64>;
@group(0) @binding(3) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_pairs { return; }
    out[i] = a[i * 2u] * b[i * 2u] + a[i * 2u + 1u] * b[i * 2u + 1u];
}
";

/// WGSL fragment: `y[i] += alpha * x[i]` (real scalar on f64 arrays).
pub const WGSL_AXPY_F64: &str = r"
struct Params { n: u32, pad0: u32, alpha: f64, }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f64>;
@group(0) @binding(2) var<storage, read_write> y: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    y[i] = y[i] + params.alpha * x[i];
}
";

/// WGSL fragment: `p[i] = x[i] + beta * p[i]` (CG direction update).
pub const WGSL_XPAY_F64: &str = r"
struct Params { n: u32, pad0: u32, beta: f64, }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f64>;
@group(0) @binding(2) var<storage, read_write> p: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    p[i] = x[i] + params.beta * p[i];
}
";

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_kernel_file_has_all_entries() {
        assert!(WGSL_CG_KERNELS_F64.contains("fn complex_dot_re"));
        assert!(WGSL_CG_KERNELS_F64.contains("fn axpy"));
        assert!(WGSL_CG_KERNELS_F64.contains("fn xpay"));
    }

    #[test]
    fn standalone_shaders_valid() {
        assert!(WGSL_COMPLEX_DOT_RE_F64.contains("n_pairs"));
        assert!(WGSL_AXPY_F64.contains("alpha"));
        assert!(WGSL_XPAY_F64.contains("beta"));
    }

    #[test]
    fn gpu_resident_cg_shaders_absorbed() {
        assert!(WGSL_SUM_REDUCE_F64.contains("fn main"));
        assert!(WGSL_CG_COMPUTE_ALPHA_F64.contains("alpha"));
        assert!(WGSL_CG_COMPUTE_BETA_F64.contains("beta"));
        assert!(WGSL_CG_UPDATE_XR_F64.contains("x[i]"));
        assert!(WGSL_CG_UPDATE_P_F64.contains("p[i]"));
    }
}
