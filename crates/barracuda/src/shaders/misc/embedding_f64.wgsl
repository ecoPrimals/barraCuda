// SPDX-License-Identifier: AGPL-3.0-only
// Embedding_f64.wgsl — Lookup table operation (f64 canonical)
//
// Pure WGSL implementation (universal compute)
// Hardware-agnostic (works on any GPU/CPU via WebGPU)

struct Params {
    num_indices: u32,
    embedding_dim: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<f64>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_indices * params.embedding_dim) {
        return;
    }

    let batch_idx = idx / params.embedding_dim;
    let embed_idx = idx % params.embedding_dim;

    let weight_idx = indices[batch_idx];
    let weight_offset = weight_idx * params.embedding_dim + embed_idx;

    output[idx] = weight[weight_offset];
}
