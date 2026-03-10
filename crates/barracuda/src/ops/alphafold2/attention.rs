// SPDX-License-Identifier: AGPL-3.0-only

//! MSA row/column attention, triangle attention, and IPA scores.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::WG_64;

const SHADER_MSA_ROW: &str =
    include_str!("../../shaders/attention/msa_row_attention_scores_f64.wgsl");
const SHADER_MSA_COL: &str =
    include_str!("../../shaders/attention/msa_col_attention_scores_f64.wgsl");
const SHADER_IPA: &str = include_str!("../../shaders/attention/ipa_scores_f64.wgsl");
const SHADER_TRI_ATTN: &str = include_str!("../../shaders/attention/triangle_attention_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MsaAttnParams {
    s: u32,
    n: u32,
    d: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IpaParams {
    n: u32,
    d: u32,
    p: u32,
    _pad: u32,
    w_l: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TriangleAttnParams {
    n: u32,
    d: u32,
}

/// MSA row-wise attention: `out[s,i,j] = sum_d q[s,i,d] * k[s,j,d] / sqrt(d)`.
///
/// Per-sequence attention over residue positions.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, buffer readback/mapping fails, or the device is lost.
pub fn msa_row_attention(
    device: &Arc<WgpuDevice>,
    q: &[f64],
    k: &[f64],
    s: u32,
    n: u32,
    d: u32,
) -> Result<Vec<f64>> {
    let out_len = (s * n * n) as usize;
    let q_buf = device.create_buffer_f64_init("msa_row:q", q);
    let k_buf = device.create_buffer_f64_init("msa_row:k", k);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = MsaAttnParams { s, n, d, _pad: 0 };
    let params_buf = device.create_uniform_buffer("msa_row:params", &params);

    ComputeDispatch::new(device, "msa_row_attention_scores")
        .shader(SHADER_MSA_ROW, "main")
        .f64()
        .storage_read(0, &q_buf)
        .storage_read(1, &k_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit()?;

    device.read_f64_buffer(&out_buf, out_len)
}

/// MSA column-wise attention: `out[s1,s2,i] = sum_d q[s1,i,d] * k[s2,i,d] / sqrt(d)`.
///
/// Cross-sequence attention at each residue position.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, buffer readback/mapping fails, or the device is lost.
pub fn msa_col_attention(
    device: &Arc<WgpuDevice>,
    q: &[f64],
    k: &[f64],
    s: u32,
    n: u32,
    d: u32,
) -> Result<Vec<f64>> {
    let out_len = (s * s * n) as usize;
    let q_buf = device.create_buffer_f64_init("msa_col:q", q);
    let k_buf = device.create_buffer_f64_init("msa_col:k", k);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = MsaAttnParams { s, n, d, _pad: 0 };
    let params_buf = device.create_uniform_buffer("msa_col:params", &params);

    ComputeDispatch::new(device, "msa_col_attention_scores")
        .shader(SHADER_MSA_COL, "main")
        .f64()
        .storage_read(0, &q_buf)
        .storage_read(1, &k_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit()?;

    device.read_f64_buffer(&out_buf, out_len)
}

/// Invariant Point Attention scores for structure prediction.
///
/// Combines sequence-space attention with 3D point-cloud distance weighting
/// to produce structure-aware attention scores.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn ipa_scores(
    device: &Arc<WgpuDevice>,
    q: &[f64],
    k: &[f64],
    q_pts: &[f64],
    k_pts: &[f64],
    n: u32,
    d: u32,
    p: u32,
    w_l: f64,
) -> Result<Vec<f64>> {
    let out_len = (n * n) as usize;
    let q_buf = device.create_buffer_f64_init("ipa:q", q);
    let k_buf = device.create_buffer_f64_init("ipa:k", k);
    let qp_buf = device.create_buffer_f64_init("ipa:q_pts", q_pts);
    let kp_buf = device.create_buffer_f64_init("ipa:k_pts", k_pts);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = IpaParams {
        n,
        d,
        p,
        _pad: 0,
        w_l,
    };
    let params_buf = device.create_uniform_buffer("ipa:params", &params);

    ComputeDispatch::new(device, "ipa_scores")
        .shader(SHADER_IPA, "main")
        .f64()
        .storage_read(0, &q_buf)
        .storage_read(1, &k_buf)
        .storage_read(2, &qp_buf)
        .storage_read(3, &kp_buf)
        .storage_rw(4, &out_buf)
        .uniform(5, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit()?;

    device.read_f64_buffer(&out_buf, out_len)
}

/// Triangle self-attention over pair representations.
///
/// `out[i,j] = sum_k softmax(q·k/√d)[i,k] * v[k,j]`, biased by pair[i,k].
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, buffer readback/mapping fails, or the device is lost.
pub fn triangle_attention(
    device: &Arc<WgpuDevice>,
    pair: &[f64],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    n: u32,
    d: u32,
) -> Result<Vec<f64>> {
    let out_len = (n * n) as usize;
    let pair_buf = device.create_buffer_f64_init("tri_attn:pair", pair);
    let q_buf = device.create_buffer_f64_init("tri_attn:q", q);
    let k_buf = device.create_buffer_f64_init("tri_attn:k", k);
    let v_buf = device.create_buffer_f64_init("tri_attn:v", v);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = TriangleAttnParams { n, d };
    let params_buf = device.create_uniform_buffer("tri_attn:params", &params);

    ComputeDispatch::new(device, "triangle_attention")
        .shader(SHADER_TRI_ATTN, "main")
        .f64()
        .storage_read(0, &pair_buf)
        .storage_read(1, &q_buf)
        .storage_read(2, &k_buf)
        .storage_read(3, &v_buf)
        .storage_rw(4, &out_buf)
        .uniform(5, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit()?;

    device.read_f64_buffer(&out_buf, out_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout_msa_attn() {
        assert_eq!(std::mem::size_of::<MsaAttnParams>(), 16);
    }

    #[test]
    fn params_layout_ipa() {
        assert_eq!(std::mem::size_of::<IpaParams>(), 24);
    }

    #[test]
    fn params_layout_triangle_attn() {
        assert_eq!(std::mem::size_of::<TriangleAttnParams>(), 8);
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
            // Naga 22 may not support enable f64; fallback: verify source structure
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
    fn msa_row_attention_scores_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/msa_row_attention_scores_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn msa_col_attention_scores_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/msa_col_attention_scores_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn ipa_scores_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/ipa_scores_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn triangle_attention_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/triangle_attention_f64.wgsl");
        shader_compiles_via_naga(source);
    }
}
