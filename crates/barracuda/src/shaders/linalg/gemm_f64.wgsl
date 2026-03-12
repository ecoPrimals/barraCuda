// SPDX-License-Identifier: AGPL-3.0-only
// Dense Matrix Multiply (f64) - Batched GEMM
//
// C = A * B  where A is [M x K], B is [K x N], C is [M x N]
// Supports batched operation: batch_size independent multiplications
//
// Memory layout: Row-major flat arrays
//   A_batch: [batch_size × M × K] f64
//   B_batch: [batch_size × K × N] f64
//   C_batch: [batch_size × M × N] f64
//
// Use cases: HFB Hamiltonian assembly, density computation, energy integrals
//
// Algorithm: Tiled multiplication with workgroup shared memory
// Tile size: 16x16 — optimal for HFB matrices (20-50 dim)
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision via SPIR-V/Vulkan
// - Zero unsafe code
// - Self-contained (no external dependencies)

struct GemmParams {
    M: u32,            // Rows of A / C
    K: u32,            // Cols of A / Rows of B
    N: u32,            // Cols of B / C
    batch_size: u32,   // Number of independent multiplications
    alpha: f64,        // Scalar: C = alpha * A * B + beta * C
    beta: f64,         // Scalar for accumulation (0.0 for pure multiply)
}

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> A_batch: array<f64>;
@group(0) @binding(2) var<storage, read> B_batch: array<f64>;
@group(0) @binding(3) var<storage, read_write> C_batch: array<f64>;

const TILE: u32 = 16u;

var<workgroup> tile_A: array<f64, 256>;  // 16 x 16
var<workgroup> tile_B: array<f64, 256>;  // 16 x 16

// Main batched GEMM kernel
// Dispatch: (ceil(N/16), ceil(M/16), batch_size) workgroups
@compute @workgroup_size(16, 16, 1)
fn gemm_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let col = global_id.x;       // Column in C
    let row = global_id.y;       // Row in C
    let batch_idx = wg_id.z;     // Batch index

    let lid_x = local_id.x;
    let lid_y = local_id.y;

    let M = params.M;
    let K = params.K;
    let N = params.N;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let a_base = batch_idx * M * K;
    let b_base = batch_idx * K * N;
    let c_base = batch_idx * M * N;

    var acc: f64 = f64(0.0);

    // Tile over K dimension
    let n_tiles = (K + TILE - 1u) / TILE;
    for (var t = 0u; t < n_tiles; t++) {
        // Load tile of A: rows [wg_id.y*TILE .. (wg_id.y+1)*TILE], cols [t*TILE .. (t+1)*TILE]
        let a_row = row;
        let a_col = t * TILE + lid_x;
        if (a_row < M && a_col < K) {
            tile_A[lid_y * TILE + lid_x] = A_batch[a_base + a_row * K + a_col];
        } else {
            tile_A[lid_y * TILE + lid_x] = f64(0.0);
        }

        // Load tile of B: rows [t*TILE .. (t+1)*TILE], cols [wg_id.x*TILE .. (wg_id.x+1)*TILE]
        let b_row = t * TILE + lid_y;
        let b_col = col;
        if (b_row < K && b_col < N) {
            tile_B[lid_y * TILE + lid_x] = B_batch[b_base + b_row * N + b_col];
        } else {
            tile_B[lid_y * TILE + lid_x] = f64(0.0);
        }

        workgroupBarrier();

        // Accumulate dot product for this tile
        for (var kk = 0u; kk < TILE; kk++) {
            acc = acc + tile_A[lid_y * TILE + kk] * tile_B[kk * TILE + lid_x];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < M && col < N) {
        let c_idx = c_base + row * N + col;
        if (params.beta != f64(0.0)) {
            C_batch[c_idx] = params.alpha * acc + params.beta * C_batch[c_idx];
        } else {
            C_batch[c_idx] = params.alpha * acc;
        }
    }
}

// Simple element-wise variant: C = A * B (Hadamard product)
// Dispatch: (ceil(total/256), 1, 1) where total = batch_size * M * N
@compute @workgroup_size(256)
fn elementwise_mul_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.M * params.N;
    if (idx >= total) {
        return;
    }
    C_batch[idx] = A_batch[idx] * B_batch[idx];
}

// Matrix-vector multiply: y = A * x (batched)
// A: [batch_size × M × K], x: [batch_size × K], y: [batch_size × M]
// Thread per (row, batch): computes dot product of A[batch][row,:] and x[batch,:]
// Dispatch: (ceil(M/256), batch_size, 1)
@compute @workgroup_size(256, 1, 1)
fn batched_matvec_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let batch_idx = global_id.y;

    let M = params.M;
    let K = params.K;

    if (row >= M || batch_idx >= params.batch_size) {
        return;
    }

    let a_base = batch_idx * M * K;
    let x_base = batch_idx * K;
    let y_base = batch_idx * M;

    var sum: f64 = f64(0.0);
    for (var k = 0u; k < K; k++) {
        sum = sum + A_batch[a_base + row * K + k] * B_batch[x_base + k];
    }

    C_batch[y_base + row] = sum;
}
