// SPDX-License-Identifier: AGPL-3.0-only
// smith_waterman_banded_f64.wgsl — Banded Smith-Waterman local alignment (f64)
//
// Local sequence alignment with affine gap penalties.
// Only cells within `band_width` of the main diagonal are computed,
// limiting memory to O(n × band_width) and enabling near-linear runtime
// for sequences of similar length.
//
// Anti-diagonal wavefront: cells on the same anti-diagonal (i+j = d) are
// independent — each depends only on diagonals d-1 and d-2.  The Rust
// wrapper submits one dispatch per anti-diagonal (d = 2..n+m).
//
// Scoring model:
//   H[i][j] = max(0,
//                 H[i-1][j-1] + subst[q[i-1]][t[j-1]],  // match/mismatch
//                 E[i][j],                                // gap in query
//                 F[i][j])                                // gap in target
//   E[i][j] = max(H[i][j-1] - gap_open, E[i][j-1] - gap_extend)
//   F[i][j] = max(H[i-1][j] - gap_open, F[i-1][j] - gap_extend)
//
// Storage: h_mat / e_mat / f_mat are (n+1)×(m+1) row-major arrays.
// Row 0 and column 0 must be zero-initialised by the Rust wrapper.
//
// Absorbed from wetSpring handoff §Shader Design 1 (Feb 2026).

// Note: f64 is not allowed in var<uniform>.  Params are passed as a storage
// read buffer so that gap_open and gap_extend can be f64.
struct SwParams {
    n:          u32,   // query length   (rows  1..n)
    m:          u32,   // target length  (cols  1..m)
    band_width: u32,   // max |row-col|; 0 = full DP (no band)
    diagonal:   u32,   // current d = row + col (swept 2..n+m by Rust)
    gap_open:   f64,   // affine gap open   (positive → subtracted)
    gap_extend: f64,   // affine gap extend (positive → subtracted)
}

@group(0) @binding(0) var<storage, read>      params: SwParams;
@group(0) @binding(1) var<storage, read>      query:  array<u32>;   // [n]   nucleotide 0..3
@group(0) @binding(2) var<storage, read>      tgt:    array<u32>;   // [m]   nucleotide 0..3
@group(0) @binding(3) var<storage, read>      subst:  array<f64>;   // [4×4] substitution scores
@group(0) @binding(4) var<storage, read_write> h_mat: array<f64>;   // [(n+1)×(m+1)]
@group(0) @binding(5) var<storage, read_write> e_mat: array<f64>;   // [(n+1)×(m+1)]
@group(0) @binding(6) var<storage, read_write> f_mat: array<f64>;   // [(n+1)×(m+1)]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d    = params.diagonal;
    let cols = params.m + 1u;   // stride for (n+1)×(m+1) storage

    // Valid row range on anti-diagonal d (1 ≤ i ≤ n, 1 ≤ j=d-i ≤ m)
    let i_lo = max(1u, select(0u, d - params.m, d > params.m));
    let i_hi = min(params.n, d - 1u);
    if (i_hi < i_lo) { return; }

    let i = i_lo + gid.x;
    if (i > i_hi) { return; }
    let j = d - i;

    // Band constraint: skip cells too far from main diagonal
    let bw = params.band_width;
    if (bw > 0u) {
        let diff = select(i - j, j - i, j > i);
        if (diff > bw) { return; }
    }

    let idx_cur  = i        * cols + j;
    let idx_diag = (i - 1u) * cols + (j - 1u);
    let idx_up   = (i - 1u) * cols + j;
    let idx_left = i        * cols + (j - 1u);

    // Substitution score for (query[i-1], target[j-1])
    let s = subst[query[i - 1u] * 4u + tgt[j - 1u]];

    // E[i][j] = gap in query (gap inserted into target, extending along target)
    let e_val = max(
        h_mat[idx_left] - params.gap_open,
        e_mat[idx_left] - params.gap_extend,
    );

    // F[i][j] = gap in target (gap inserted into query, extending along query)
    let f_val = max(
        h_mat[idx_up] - params.gap_open,
        f_mat[idx_up] - params.gap_extend,
    );

    // H[i][j] — best alignment score ending here; 0 = local alignment restart.
    // Use f64(0.0) for zero: abstract float literal 0.0 resolves to f32 in naga.
    let best_gap   = max(e_val, f_val);
    let sub_score  = h_mat[idx_diag] + s;
    let best_score = max(sub_score, best_gap);
    var h_val: f64 = f64(0.0);
    if best_score > f64(0.0) {
        h_val = best_score;
    }

    e_mat[idx_cur] = e_val;
    f_mat[idx_cur] = f_val;
    h_mat[idx_cur] = h_val;
}
