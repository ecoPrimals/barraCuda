// SPDX-License-Identifier: AGPL-3.0-or-later
// snp_calling_f64.wgsl — Position-Parallel SNP Calling
//
// One thread per alignment position. Each thread counts allele
// frequencies across all sequences at its column, determines the
// reference (most common) allele, and reports whether the position
// is polymorphic.
//
// GPU dispatch: ceil(alignment_length / 256) workgroups, 256 threads each.
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct SnpParams {
    alignment_length: u32,
    n_sequences:      u32,
    min_depth:        u32,
    _pad:             u32,
}

@group(0) @binding(0) var<uniform>             params:       SnpParams;
// Bases encoded as u32: A=0, C=1, G=2, T=3, gap/N=4
@group(0) @binding(1) var<storage, read>       sequences:    array<u32>;  // [n_sequences * alignment_length]
@group(0) @binding(2) var<storage, read_write> is_variant:   array<u32>;  // [alignment_length] 0/1
@group(0) @binding(3) var<storage, read_write> ref_allele:   array<u32>;  // [alignment_length]
@group(0) @binding(4) var<storage, read_write> depth_out:    array<u32>;  // [alignment_length]
@group(0) @binding(5) var<storage, read_write> alt_freq_out: array<f64>;  // [alignment_length]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    if pos >= params.alignment_length { return; }

    var counts: array<u32, 4> = array<u32, 4>(0u, 0u, 0u, 0u);
    var depth: u32 = 0u;

    for (var s: u32 = 0u; s < params.n_sequences; s = s + 1u) {
        let base = sequences[s * params.alignment_length + pos];
        if base < 4u {
            counts[base] = counts[base] + 1u;
            depth = depth + 1u;
        }
    }

    depth_out[pos] = depth;

    var ref_idx: u32 = 0u;
    var ref_count: u32 = counts[0];
    for (var i: u32 = 1u; i < 4u; i = i + 1u) {
        if counts[i] > ref_count {
            ref_count = counts[i];
            ref_idx = i;
        }
    }
    ref_allele[pos] = ref_idx;

    var n_alleles: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if counts[i] > 0u {
            n_alleles = n_alleles + 1u;
        }
    }

    if n_alleles >= 2u && depth >= params.min_depth {
        is_variant[pos] = 1u;
        let alt_count = depth - ref_count;
        alt_freq_out[pos] = f64(alt_count) / f64(depth);
    } else {
        is_variant[pos] = 0u;
        alt_freq_out[pos] = f64(0);
    }
}
