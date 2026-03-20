// SPDX-License-Identifier: AGPL-3.0-or-later
// Tiled Matrix Multiplication - Memory-Optimized (f64 canonical)
//
// **OPTIMIZATION**: Shared memory tiling for 70-80% bandwidth utilization
//
// C = A * B where A is (M, K), B is (K, N), C is (M, N)

@group(0) @binding(0) var<storage, read> A: array<f64>;
@group(0) @binding(1) var<storage, read> B: array<f64>;
@group(0) @binding(2) var<storage, read_write> C: array<f64>;

struct MatmulParams {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(3) var<uniform> params: MatmulParams;

// Shared memory tiles (16x16 = 256 floats per tile)
var<workgroup> tileA: array<f64, 256>;  // 16x16 tile of A
var<workgroup> tileB: array<f64, 256>;  // 16x16 tile of B

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;

    let local_row = local_id.y;
    let local_col = local_id.x;

    let in_bounds = row < params.M && col < params.N;

    var sum = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        let a_row = row;
        let a_col = tile * TILE_SIZE + local_col;

        if (a_row < params.M && a_col < params.K) {
            tileA[local_row * TILE_SIZE + local_col] = A[a_row * params.K + a_col];
        } else {
            tileA[local_row * TILE_SIZE + local_col] = 0.0;
        }

        let b_row = tile * TILE_SIZE + local_row;
        let b_col = col;

        if (b_row < params.K && b_col < params.N) {
            tileB[local_row * TILE_SIZE + local_col] = B[b_row * params.N + b_col];
        } else {
            tileB[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            let a_val = tileA[local_row * TILE_SIZE + k];
            let b_val = tileB[k * TILE_SIZE + local_col];
            sum = sum + a_val * b_val;
        }

        workgroupBarrier();
    }

    if (in_bounds) {
        C[row * params.N + col] = sum;
    }
}
