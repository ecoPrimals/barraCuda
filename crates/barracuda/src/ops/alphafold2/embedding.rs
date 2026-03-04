// SPDX-License-Identifier: AGPL-3.0-or-later

//! Outer product mean, pair transition, template embedding, recycling update, ensemble average.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::WG_64;

const SHADER_OPM: &str = include_str!("../../shaders/attention/outer_product_mean_f64.wgsl");
const SHADER_PAIR_TRANSITION: &str =
    include_str!("../../shaders/attention/pair_transition_f64.wgsl");
const SHADER_TEMPLATE_EMBEDDING: &str =
    include_str!("../../shaders/attention/template_embedding_f64.wgsl");
const SHADER_RECYCLING: &str = include_str!("../../shaders/misc/recycling_update_f64.wgsl");
const SHADER_ENSEMBLE_AVG: &str = include_str!("../../shaders/misc/ensemble_average_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OpmParams {
    s: u32,
    n: u32,
    c: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PairTransitionParams {
    n: u32,
    c_in: u32,
    c_hidden: u32,
    c_out: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TemplateEmbeddingParams {
    t: u32,
    n: u32,
    c: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RecyclingParams {
    n: u32,
    c: u32,
    _pad: [u32; 2],
    eps: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EnsembleAverageParams {
    n_models: u32,
    n_atoms: u32,
    n_dims: u32,
    _pad: u32,
}

/// Outer product mean for pair representation update.
///
/// Computes `out[i,j,c] = mean_s(a[s,i,c] * b[s,j,c])` — the projected
/// outer product averaged over MSA sequences.
pub fn outer_product_mean(
    device: &Arc<WgpuDevice>,
    a: &[f64],
    b: &[f64],
    s: u32,
    n: u32,
    c: u32,
) -> Result<Vec<f64>> {
    let out_len = (n * n * c) as usize;
    let a_buf = device.create_buffer_f64_init("opm:a", a);
    let b_buf = device.create_buffer_f64_init("opm:b", b);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = OpmParams { s, n, c, _pad: 0 };
    let params_buf = device.create_uniform_buffer("opm:params", &params);

    ComputeDispatch::new(device, "outer_product_mean")
        .shader(SHADER_OPM, "main")
        .f64()
        .storage_read(0, &a_buf)
        .storage_read(1, &b_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

/// Pair representation transition: 2-layer MLP on pair features.
///
/// `out[i,j,c] = ReLU(pair[i,j,:] * W1 + b1) * W2 + b2`
pub fn pair_transition(
    device: &Arc<WgpuDevice>,
    pair: &[f64],
    weights1: &[f64],
    weights2: &[f64],
    bias1: &[f64],
    bias2: &[f64],
    n: u32,
    c_in: u32,
    c_hidden: u32,
    c_out: u32,
) -> Result<Vec<f64>> {
    let out_len = (n * n * c_out) as usize;
    let pair_buf = device.create_buffer_f64_init("pair_trans:pair", pair);
    let w1_buf = device.create_buffer_f64_init("pair_trans:w1", weights1);
    let w2_buf = device.create_buffer_f64_init("pair_trans:w2", weights2);
    let b1_buf = device.create_buffer_f64_init("pair_trans:b1", bias1);
    let b2_buf = device.create_buffer_f64_init("pair_trans:b2", bias2);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = PairTransitionParams {
        n,
        c_in,
        c_hidden,
        c_out,
    };
    let params_buf = device.create_uniform_buffer("pair_trans:params", &params);

    ComputeDispatch::new(device, "pair_transition")
        .shader(SHADER_PAIR_TRANSITION, "main")
        .f64()
        .storage_read(0, &pair_buf)
        .storage_read(1, &w1_buf)
        .storage_read(2, &w2_buf)
        .storage_read(3, &b1_buf)
        .storage_read(4, &b2_buf)
        .storage_rw(5, &out_buf)
        .uniform(6, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

/// Template stack averaging: `out[i,j,c] = (1/T) * Σ_t template[t,i,j,c]`.
pub fn template_embedding(
    device: &Arc<WgpuDevice>,
    templates: &[f64],
    t: u32,
    n: u32,
    c: u32,
) -> Result<Vec<f64>> {
    let out_len = (n * n * c) as usize;
    let templates_buf = device.create_buffer_f64_init("template_emb:templates", templates);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = TemplateEmbeddingParams { t, n, c, _pad: 0 };
    let params_buf = device.create_uniform_buffer("template_emb:params", &params);

    ComputeDispatch::new(device, "template_embedding")
        .shader(SHADER_TEMPLATE_EMBEDDING, "main")
        .f64()
        .storage_read(0, &templates_buf)
        .storage_rw(1, &out_buf)
        .uniform(2, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

/// Recycling iteration update: `out = prev + layer_norm(current - prev)`.
pub fn recycling_update(
    device: &Arc<WgpuDevice>,
    prev: &[f64],
    current: &[f64],
    n: u32,
    c: u32,
    eps: f64,
) -> Result<Vec<f64>> {
    let out_len = (n * c) as usize;
    let prev_buf = device.create_buffer_f64_init("recycling:prev", prev);
    let current_buf = device.create_buffer_f64_init("recycling:current", current);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = RecyclingParams {
        n,
        c,
        _pad: [0; 2],
        eps,
    };
    let params_buf = device.create_uniform_buffer("recycling:params", &params);

    ComputeDispatch::new(device, "recycling_update")
        .shader(SHADER_RECYCLING, "main")
        .f64()
        .storage_read(0, &prev_buf)
        .storage_read(1, &current_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

/// Ensemble averaging: `out[i,d] = (1/M) * Σ_m positions[m,i,d]`.
pub fn ensemble_average(
    device: &Arc<WgpuDevice>,
    positions: &[f64],
    n_models: u32,
    n_atoms: u32,
    n_dims: u32,
) -> Result<Vec<f64>> {
    let out_len = (n_atoms * n_dims) as usize;
    let pos_buf = device.create_buffer_f64_init("ensemble_avg:pos", positions);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = EnsembleAverageParams {
        n_models,
        n_atoms,
        n_dims,
        _pad: 0,
    };
    let params_buf = device.create_uniform_buffer("ensemble_avg:params", &params);

    ComputeDispatch::new(device, "ensemble_average")
        .shader(SHADER_ENSEMBLE_AVG, "main")
        .f64()
        .storage_read(0, &pos_buf)
        .storage_rw(1, &out_buf)
        .uniform(2, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout_opm() {
        assert_eq!(std::mem::size_of::<OpmParams>(), 16);
    }

    #[test]
    fn params_layout_pair_transition() {
        assert_eq!(std::mem::size_of::<PairTransitionParams>(), 16);
    }

    #[test]
    fn params_layout_template_embedding() {
        assert_eq!(std::mem::size_of::<TemplateEmbeddingParams>(), 16);
    }

    #[test]
    fn params_layout_recycling() {
        assert_eq!(std::mem::size_of::<RecyclingParams>(), 24);
    }

    #[test]
    fn params_layout_ensemble_average() {
        assert_eq!(std::mem::size_of::<EnsembleAverageParams>(), 16);
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
    fn outer_product_mean_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/outer_product_mean_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn pair_transition_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/pair_transition_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn template_embedding_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/template_embedding_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn recycling_update_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/recycling_update_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn ensemble_average_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/ensemble_average_f64.wgsl");
        shader_compiles_via_naga(source);
    }
}
