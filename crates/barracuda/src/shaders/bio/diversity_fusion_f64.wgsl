// SPDX-License-Identifier: AGPL-3.0-only
//
// diversity_fusion_f64.wgsl — fused Shannon + Simpson + evenness in single dispatch
//
// wetSpring extension for ToadStool absorption (Write phase).
// Computes three diversity metrics in one kernel pass over abundance data,
// avoiding three separate FusedMapReduceF64 dispatches.
//
// Binding layout:
//   @group(0) @binding(0) uniform  { n_samples: u32, n_species: u32 }
//   @group(0) @binding(1) storage  read   abundances[n_samples * n_species]  (f64)
//   @group(0) @binding(2) storage  r/w    results[n_samples * 3]             (f64)
//
// Results layout per sample (3 f64 values):
//   [0] Shannon entropy  H' = -Σ pᵢ ln(pᵢ)        (where pᵢ > 0)
//   [1] Simpson index    D  = 1 - Σ pᵢ²
//   [2] Pielou evenness  J' = H' / ln(S_obs)        (where S_obs = species with pᵢ > 0)
//
// Dispatch: ceil(n_samples / 64) workgroups × 1 × 1
//   Each thread processes one sample (all species for that sample).
//   @workgroup_size(64)

struct Params {
    n_samples: u32,
    n_species: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> abundances: array<f64>;
@group(0) @binding(2) var<storage, read_write> results: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample_idx = gid.x;
    if (sample_idx >= params.n_samples) {
        return;
    }

    let base = sample_idx * params.n_species;
    let out_base = sample_idx * 3u;

    // First pass: compute total abundance for normalization
    var total: f64 = 0.0lf;
    for (var i = 0u; i < params.n_species; i = i + 1u) {
        total = total + abundances[base + i];
    }

    // Guard against empty samples
    if (total <= 0.0lf) {
        results[out_base] = 0.0lf;
        results[out_base + 1u] = 0.0lf;
        results[out_base + 2u] = 0.0lf;
        return;
    }

    // Second pass: fused Shannon + Simpson + observed species count
    var shannon: f64 = 0.0lf;
    var simpson_sum: f64 = 0.0lf;
    var s_obs: f64 = 0.0lf;

    for (var i = 0u; i < params.n_species; i = i + 1u) {
        let count = abundances[base + i];
        if (count > 0.0lf) {
            let p = count / total;
            shannon = shannon - p * log_f64(p);
            simpson_sum = simpson_sum + p * p;
            s_obs = s_obs + 1.0lf;
        }
    }

    results[out_base] = shannon;
    results[out_base + 1u] = 1.0lf - simpson_sum;

    // Pielou evenness: H' / ln(S_obs), undefined for S_obs <= 1
    if (s_obs > 1.0lf) {
        results[out_base + 2u] = shannon / log_f64(s_obs);
    } else {
        results[out_base + 2u] = 0.0lf;
    }
}
