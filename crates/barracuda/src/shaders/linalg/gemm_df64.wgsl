// SPDX-License-Identifier: AGPL-3.0-or-later
// Dense Matrix Multiply (DF64) — Hybrid Core Streaming GEMM
//
// Prepend: df64_core.wgsl
//
// C = alpha * A * B + beta * C
// A: [batch × M × K], B: [batch × K × N], C: [batch × M × N]
//
// HYBRID PRECISION:
//   DF64 (FP32 cores): entire K-dimension dot product accumulation
//   f64  (FP64 units): global memory load/store, alpha/beta scaling
//
// Shared memory stores (hi, lo) f32 pairs in separate arrays.
// The inner accumulation loop is 100% FP32 arithmetic.
//
// Tile size: 16×16 (same as f64 variant)

struct GemmParams {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
    alpha: f64,
    beta: f64,
    trans_a: u32,
    trans_b: u32,
}

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> A_batch: array<f64>;
@group(0) @binding(2) var<storage, read> B_batch: array<f64>;
@group(0) @binding(3) var<storage, read_write> C_batch: array<f64>;

const TILE: u32 = 16u;

var<workgroup> tile_A_hi: array<f32, 256>;
var<workgroup> tile_A_lo: array<f32, 256>;
var<workgroup> tile_B_hi: array<f32, 256>;
var<workgroup> tile_B_lo: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn gemm_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let col = global_id.x;
    let row = global_id.y;
    let batch_idx = wg_id.z;

    let lid_x = local_id.x;
    let lid_y = local_id.y;

    let M = params.M;
    let K = params.K;
    let N = params.N;

    if (batch_idx >= params.batch_size) { return; }

    let a_base = batch_idx * M * K;
    let b_base = batch_idx * K * N;
    let c_base = batch_idx * M * N;

    // DF64 accumulator — all FP32 arithmetic from here
    var acc = df64_zero();

    let n_tiles = (K + TILE - 1u) / TILE;
    for (var t = 0u; t < n_tiles; t++) {
        // Load tile of A: f64 → (hi, lo) f32 pair
        let a_row = row;
        let a_col = t * TILE + lid_x;
        if (a_row < M && a_col < K) {
            let val = A_batch[a_base + a_row * K + a_col];
            let hi = f32(val);
            let lo = f32(val - f64(hi));
            tile_A_hi[lid_y * TILE + lid_x] = hi;
            tile_A_lo[lid_y * TILE + lid_x] = lo;
        } else {
            tile_A_hi[lid_y * TILE + lid_x] = 0.0;
            tile_A_lo[lid_y * TILE + lid_x] = 0.0;
        }

        // Load tile of B: f64 → (hi, lo) f32 pair
        let b_row = t * TILE + lid_y;
        let b_col = col;
        if (b_row < K && b_col < N) {
            let val = B_batch[b_base + b_row * N + b_col];
            let hi = f32(val);
            let lo = f32(val - f64(hi));
            tile_B_hi[lid_y * TILE + lid_x] = hi;
            tile_B_lo[lid_y * TILE + lid_x] = lo;
        } else {
            tile_B_hi[lid_y * TILE + lid_x] = 0.0;
            tile_B_lo[lid_y * TILE + lid_x] = 0.0;
        }

        workgroupBarrier();

        // DF64 dot product — pure FP32 arithmetic on FP32 cores
        for (var kk = 0u; kk < TILE; kk++) {
            let a_df = Df64(tile_A_hi[lid_y * TILE + kk], tile_A_lo[lid_y * TILE + kk]);
            let b_df = Df64(tile_B_hi[kk * TILE + lid_x], tile_B_lo[kk * TILE + lid_x]);
            acc = df64_add(acc, df64_mul(a_df, b_df));
        }

        workgroupBarrier();
    }

    // Boundary: DF64 → f64 for output
    if (row < M && col < N) {
        let c_idx = c_base + row * N + col;
        let result = df64_to_f64(acc);
        if (params.beta != f64(0.0)) {
            C_batch[c_idx] = params.alpha * result + params.beta * C_batch[c_idx];
        } else {
            C_batch[c_idx] = params.alpha * result;
        }
    }
}

@compute @workgroup_size(256)
fn elementwise_mul_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.M * params.N;
    if (idx >= total) { return; }
    C_batch[idx] = A_batch[idx] * B_batch[idx];
}

@compute @workgroup_size(256, 1, 1)
fn batched_matvec_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let batch_idx = global_id.y;

    let M = params.M;
    let K = params.K;

    if (row >= M || batch_idx >= params.batch_size) { return; }

    let a_base = batch_idx * M * K;
    let x_base = batch_idx * K;
    let y_base = batch_idx * M;

    var sum = df64_zero();
    for (var k = 0u; k < K; k++) {
        let a_val = A_batch[a_base + row * K + k];
        let b_val = B_batch[x_base + k];
        sum = df64_add(sum, df64_mul(df64_from_f64(a_val), df64_from_f64(b_val)));
    }

    C_batch[y_base + row] = df64_to_f64(sum);
}
