// SPDX-License-Identifier: AGPL-3.0-or-later
// Bipartition encoding: convert tree bipartition strings to u32 bit-vectors
// for Robinson-Foulds distance computation.
//
// Each thread processes one bipartition. The input is a flat array of
// taxon membership flags (0 or 1) for each bipartition × n_taxa.
// The output is a packed u32 bit-vector per bipartition.
//
// Provenance: wetSpring V105 → barraCuda absorption (Mar 2026)

struct Config {
    n_bipartitions: u32,
    n_taxa: u32,
    words_per_bipart: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> membership: array<u32>;
@group(0) @binding(2) var<storage, read_write> bitvectors: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bp_idx = gid.x;
    if bp_idx >= config.n_bipartitions { return; }

    let in_base = bp_idx * config.n_taxa;
    let out_base = bp_idx * config.words_per_bipart;

    for (var w: u32 = 0u; w < config.words_per_bipart; w = w + 1u) {
        var word: u32 = 0u;
        let bit_start = w * 32u;
        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            let taxon = bit_start + b;
            if taxon < config.n_taxa {
                let flag = membership[in_base + taxon];
                word = word | (flag << b);
            }
        }
        bitvectors[out_base + w] = word;
    }
}
