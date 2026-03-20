// SPDX-License-Identifier: AGPL-3.0-or-later
// dada2_e_step.wgsl — Batch log_p_error for DADA2 E-step
//
// One GPU thread per (sequence, center) pair.
// Each thread sums precomputed log(err[from][to][qual]) over all positions.
// No GPU transcendentals — log values precomputed on CPU.
//
// Input:
//   - bases: 1 u32 per base (0=A, 1=C, 2=G, 3=T), padded to max_len
//   - quals: 1 u32 per position (phred score 0-41), padded to max_len
//   - lengths: per-sequence actual lengths
//   - center_indices: which sequences are current centers
//   - log_err: precomputed ln(err[from][to][qual]), 4×4×42 = 672 f64 values
//
// Output: n_seqs × n_centers f64 matrix of log-probabilities
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct Params {
    n_seqs: u32,
    n_centers: u32,
    max_len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bases: array<u32>;
@group(0) @binding(2) var<storage, read> quals: array<u32>;
@group(0) @binding(3) var<storage, read> lengths: array<u32>;
@group(0) @binding(4) var<storage, read> center_indices: array<u32>;
@group(0) @binding(5) var<storage, read> log_err: array<f64>;
@group(0) @binding(6) var<storage, read_write> scores: array<f64>;

@compute @workgroup_size(256)
fn e_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let seq_idx = pair_idx / params.n_centers;
    let center_slot = pair_idx % params.n_centers;

    if (seq_idx >= params.n_seqs) {
        return;
    }

    let center_idx = center_indices[center_slot];
    let seq_len = lengths[seq_idx];
    let center_len = lengths[center_idx];
    let len = min(seq_len, center_len);

    var log_p: f64 = 0.0lf;
    for (var i: u32 = 0u; i < len; i++) {
        let from_base = bases[center_idx * params.max_len + i];
        let to_base = bases[seq_idx * params.max_len + i];
        let q = quals[seq_idx * params.max_len + i];

        // log_err layout: err[from][to][q] = log_err[from * 168 + to * 42 + q]
        let err_idx = from_base * 168u + to_base * 42u + q;
        log_p += log_err[err_idx];
    }

    scores[pair_idx] = log_p;
}
