// Scatter ND - N-dimensional scatter operation with multi-dim values (f64 canonical)
//
// Scatters values into an output tensor at positions specified by indices.
// When index_rank < input_rank, scatters slices (not just scalars).
//
// Values shape: [batch_size, num_indices] + input_shape[index_rank:]
//
// Example: input [5, 4], indices [[1], [3]], values [[a,b,c,d], [e,f,g,h]]
//   → scatters values[0,:] into input[1,:] and values[1,:] into input[3,:]
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)
//
// Note: For overlapping indices, last write wins (no atomic operations).
// This is consistent with PyTorch scatter_ behavior.

struct Params {
    input_size: u32,
    indices_size: u32,
    values_size: u32,
    input_rank: u32,
    indices_rank: u32,
    index_rank: u32,     // How many leading dims each index specifies
    batch_size: u32,
    num_indices: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<f64>;
@group(0) @binding(2) var<storage, read> indices: array<f64>;
@group(0) @binding(3) var<storage, read_write> input: array<f64>;
@group(0) @binding(4) var<storage, read> input_shape: array<u32>;
@group(0) @binding(5) var<storage, read> indices_shape: array<u32>;
@group(0) @binding(6) var<storage, read> values_shape: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.values_size) {
        return;
    }

    // Compute values rank: batch + num_indices + trailing dims
    let trailing_dims = select(0u, params.input_rank - params.index_rank, params.index_rank < params.input_rank);
    let values_rank = 2u + trailing_dims;

    // Decompose flat values index into coordinates (row-major)
    var val_coords: array<u32, 8> = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var temp_idx = idx;
    for (var i: u32 = 0u; i < values_rank && i < 8u; i = i + 1u) {
        let d = values_rank - 1u - i;
        let dim_size = values_shape[d];
        val_coords[d] = temp_idx % dim_size;
        temp_idx = temp_idx / dim_size;
    }

    let batch_idx = val_coords[0];
    let index_idx = val_coords[1];

    // Read the multi-dimensional index for this scatter point
    let indices_offset = batch_idx * params.num_indices * params.index_rank
                       + index_idx * params.index_rank;

    // Compute input strides (row-major)
    var input_strides: array<u32, 8> = array<u32, 8>(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u);
    if (params.input_rank > 1u) {
        for (var i: i32 = i32(params.input_rank) - 2; i >= 0; i = i - 1) {
            let ui = u32(i);
            input_strides[ui] = input_strides[ui + 1u] * input_shape[ui + 1u];
        }
    }

    // Compute base input index from index coordinates
    var input_idx: u32 = 0u;
    for (var i: u32 = 0u; i < params.index_rank && i < 8u; i = i + 1u) {
        let coord = u32(indices[indices_offset + i]);
        // Bounds-clamp the index coordinate
        let clamped = min(coord, input_shape[i] - 1u);
        input_idx = input_idx + clamped * input_strides[i];
    }

    // Add trailing dimensions from value coordinates
    // val_coords[2..] correspond to input dimensions [index_rank..]
    for (var t: u32 = 0u; t < trailing_dims && t < 6u; t = t + 1u) {
        let input_dim = params.index_rank + t;
        let val_coord_idx = 2u + t;
        let coord = val_coords[val_coord_idx];
        input_idx = input_idx + coord * input_strides[input_dim];
    }

    // Bounds check and scatter
    if (input_idx < params.input_size) {
        input[input_idx] = values[idx]; // Last write wins for overlapping indices
    }
}
