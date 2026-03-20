// SPDX-License-Identifier: AGPL-3.0-or-later
// batch_pair_reduce_f64.wgsl — Generic O(N²) pairwise batch reduction (f64)
//
// **WetSpring handoff absorption** — DADA2 E-step, BrayCurtis, spectral match.
//
// Computes a scalar reduction over pairs (i, j) from two feature matrices:
//   A: [B × N × D]  — B batches, N items, D features each
//   B: [B × M × D]  — B batches, M items, D features each
//   out: [B × N × M] — one scalar per (batch, i, j) pair
//
// Supported operations (via `op` field in params):
//   0 = dot product:       Σ_d  A[b,i,d] · B[b,j,d]
//   1 = squared L2:        Σ_d (A[b,i,d] - B[b,j,d])²
//   2 = L1 distance:       Σ_d |A[b,i,d] - B[b,j,d]|
//   3 = log-sum-exp diff:  Σ_d  log(A[b,i,d]) - log(B[b,j,d])  (for DADA2)
//
// Thread assignment: one thread per (b, i, j) triple.
// Workgroup: (16, 16, 1) — x → i-axis, y → j-axis; outer batch loop in thread.
// For B=1 (most wetSpring use cases) this reduces to an (N×M) problem.
//
// Bindings:
//   0: config uniform
//   1: mat_a   [B × N × D] f64
//   2: mat_b   [B × M × D] f64
//   3: out     [B × N × M] f64

// f64 is enabled by compile_shader_f64() preamble injection — do not use `enable f64;`

struct PairReduceConfig {
    n_batches: u32,
    n_a:       u32,   // N
    n_b:       u32,   // M
    n_features: u32,  // D
    op:        u32,   // Operation code (see above)
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

@group(0) @binding(0) var<uniform>             config: PairReduceConfig;
@group(0) @binding(1) var<storage, read>       mat_a:  array<f64>;
@group(0) @binding(2) var<storage, read>       mat_b:  array<f64>;
@group(0) @binding(3) var<storage, read_write> out:    array<f64>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id:    vec3<u32>,
    @builtin(workgroup_id)         workgroup_id: vec3<u32>,
) {
    let i = global_id.x;   // index into A rows
    let j = global_id.y;   // index into B rows

    if (i >= config.n_a || j >= config.n_b) { return; }

    let D = config.n_features;
    let op = config.op;

    // Outer batch loop (usually B=1 for wetSpring)
    for (var b = 0u; b < config.n_batches; b = b + 1u) {
        let base_a = (b * config.n_a + i) * D;
        let base_b = (b * config.n_b + j) * D;

        var acc: f64 = f64(0.0);

        for (var d = 0u; d < D; d = d + 1u) {
            let a = mat_a[base_a + d];
            let bv = mat_b[base_b + d];

            if (op == 0u) {
                // Dot product — use explicit mul-add (fma f64 not universally supported in Naga)
                acc = a * bv + acc;
            } else if (op == 1u) {
                // Squared L2 — use explicit mul-add
                let diff = a - bv;
                acc = diff * diff + acc;
            } else if (op == 2u) {
                // L1 distance
                acc = acc + abs(a - bv);
            } else {
                // op == 3: log-sum-exp diff (DADA2 error model)
                // Σ log(a) - log(b) treated as: Σ log(a/b)
                if (a > 0.0 && bv > 0.0) {
                    acc = acc + log(a / bv);
                }
            }
        }

        out[(b * config.n_a + i) * config.n_b + j] = acc;
    }
}
