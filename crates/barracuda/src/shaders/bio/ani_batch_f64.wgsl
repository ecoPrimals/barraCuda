// ani_batch_f64.wgsl — Batch Pairwise ANI (Average Nucleotide Identity)
//
// One thread per sequence pair. Each thread walks the alignment
// positionally, counting identical non-gap/non-N bases, then divides
// to get ANI ∈ [0, 1].
//
// GPU dispatch: ceil(n_pairs / 256) workgroups, 256 threads each.
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct AniParams {
    n_pairs:     u32,
    max_seq_len: u32,
}

@group(0) @binding(0) var<uniform>             params:  AniParams;
// Sequences packed as u32-per-base: A=0, C=1, G=2, T=3, gap=4, N=5
@group(0) @binding(1) var<storage, read>       seq_a:   array<u32>;  // [n_pairs * max_seq_len]
@group(0) @binding(2) var<storage, read>       seq_b:   array<u32>;  // [n_pairs * max_seq_len]
@group(0) @binding(3) var<storage, read_write> ani_out: array<f64>;  // [n_pairs]
@group(0) @binding(4) var<storage, read_write> aligned_out: array<u32>; // [n_pairs]
@group(0) @binding(5) var<storage, read_write> identical_out: array<u32>; // [n_pairs]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if pair_idx >= params.n_pairs { return; }

    let base = pair_idx * params.max_seq_len;
    var aligned: u32 = 0u;
    var identical: u32 = 0u;

    for (var i: u32 = 0u; i < params.max_seq_len; i = i + 1u) {
        let a = seq_a[base + i];
        let b = seq_b[base + i];

        if a >= 4u || b >= 4u { continue; }

        aligned = aligned + 1u;
        if a == b {
            identical = identical + 1u;
        }
    }

    aligned_out[pair_idx] = aligned;
    identical_out[pair_idx] = identical;

    if aligned > 0u {
        ani_out[pair_idx] = f64(identical) / f64(aligned);
    } else {
        ani_out[pair_idx] = f64(0);
    }
}
