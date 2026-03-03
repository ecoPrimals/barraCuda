// Transpose: N-Dimensional transpose with arbitrary dimension permutations (f64 canonical)
// Supports both 2D transpose (swap last two dims) and N-D with permutation
//
// Algorithm: Generalized stride computation for N-D tensors
// For 2D: optimized tiled transpose
// For N-D: use permutation mapping with strides

struct Params {
    total_size: u32,
    num_dims: u32,
    is_2d: u32,        // 1 if 2D, 0 if N-D
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

// For 2D case
struct Params2D {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params_2d: Params2D;

// For N-D case
@group(0) @binding(4) var<storage, read> input_shape: array<u32>;
@group(0) @binding(5) var<storage, read> output_shape: array<u32>;
@group(0) @binding(6) var<storage, read> permutation: array<u32>;
@group(0) @binding(7) var<storage, read> input_strides: array<u32>;
@group(0) @binding(8) var<storage, read> output_strides: array<u32>;

// Shared memory tile for 2D transpose (16x16 with padding)
var<workgroup> tile: array<f64, 272>;  // 16x17 = 272 (extra column for padding)

fn tile_index(row: u32, col: u32) -> u32 {
    return row * 17u + col;  // 17 to avoid bank conflicts
}

// 2D optimized transpose
@compute @workgroup_size(16, 16)
fn main_2d(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tx = local_id.x;
    let ty = local_id.y;

    // Input position (reading)
    let input_col = workgroup_id.x * 16u + tx;
    let input_row = workgroup_id.y * 16u + ty;

    // Load tile from input (coalesced)
    if (input_row < params_2d.rows && input_col < params_2d.cols) {
        let input_idx = input_row * params_2d.cols + input_col;
        tile[tile_index(ty, tx)] = input[input_idx];
    }
    workgroupBarrier();

    // Output position (writing transposed)
    let output_col = workgroup_id.y * 16u + tx;
    let output_row = workgroup_id.x * 16u + ty;

    // Write tile to output (coalesced, transposed)
    if (output_row < params_2d.cols && output_col < params_2d.rows) {
        let output_idx = output_row * params_2d.rows + output_col;
        output[output_idx] = tile[tile_index(tx, ty)];  // Note: tx and ty swapped
    }
}

// N-D generalized transpose
@compute @workgroup_size(256)
fn main_nd(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.total_size) {
        return;
    }

    // Compute multi-dimensional output indices
    var temp_idx = out_idx;
    var in_idx = 0u;

    for (var i = 0u; i < params.num_dims; i++) {
        // Get coordinate in output space
        let out_coord = temp_idx / output_strides[i];
        temp_idx = temp_idx % output_strides[i];

        // Map to input space using permutation
        let in_dim = permutation[i];
        in_idx += out_coord * input_strides[in_dim];
    }

    output[out_idx] = input[in_idx];
}
