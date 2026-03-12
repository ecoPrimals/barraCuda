// SPDX-License-Identifier: AGPL-3.0-only
// kmer_histogram.wgsl — GPU k-mer histogram accumulation
//
// wetSpring absorption: metagenomics k-mer counting.
//
// Computes 4^k histogram from encoded k-mer sequences.
// One thread per k-mer, atomic add into histogram buffer.
//
// CPU reference: wetspring_barracuda::bio::kmer::count_kmers
//
// Bindings:
//   0: config  uniform { n_kmers, k, _pad0, _pad1 }
//   1: kmers   [n_kmers] u32 — encoded k-mer hashes
//   2: histogram [4^k] atomic<u32> — output histogram

struct KmerConfig {
    n_kmers: u32,
    k:       u32,
    _pad0:   u32,
    _pad1:   u32,
}

@group(0) @binding(0) var<uniform>             config:    KmerConfig;
@group(0) @binding(1) var<storage, read>       kmers:     array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn kmer_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= config.n_kmers {
        return;
    }

    let kmer_hash = kmers[idx];
    atomicAdd(&histogram[kmer_hash], 1u);
}
