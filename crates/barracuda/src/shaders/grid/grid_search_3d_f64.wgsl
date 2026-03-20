// SPDX-License-Identifier: AGPL-3.0-or-later
//
// grid_search_3d_f64.wgsl — 3D brute-force grid search
//
// Evaluates a scalar function f(x,y,z) at each grid point (values pre-populated by host
// or prior shader), finds global minimum via workgroup reduction.
//
// Each thread scans one slice (fixed z), finds local min; workgroup reduction finds
// global minimum across slices.
//
// Provenance: groundSpring → ToadStool absorption

enable f64;

struct GridSearchParams {
    nx: u32,
    ny: u32,
    nz: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x_grid: array<f64>;           // [NX]
@group(0) @binding(1) var<storage, read> y_grid: array<f64>;           // [NY]
@group(0) @binding(2) var<storage, read> z_grid: array<f64>;           // [NZ]
@group(0) @binding(3) var<storage, read_write> values: array<f64>;     // [NX*NY*NZ]
@group(0) @binding(4) var<storage, read_write> out_min_val: array<f64>;  // [1]
@group(0) @binding(5) var<storage, read_write> out_min_idx: array<u32>;   // [3] -> ix, iy, iz
@group(0) @binding(6) var<uniform> params: GridSearchParams;

var<workgroup> shared_min_val: array<f64, 256>;
var<workgroup> shared_min_ix: array<u32, 256>;
var<workgroup> shared_min_iy: array<u32, 256>;
var<workgroup> shared_min_iz: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let nx = params.nx;
    let ny = params.ny;
    let nz = params.nz;
    let n_total = nx * ny * nz;
    let tid = lid.x;

    // Each thread scans its chunk of the 3D grid (stride by workgroup size)
    var local_min_val: f64 = 1e308;
    var local_min_ix: u32 = 0u;
    var local_min_iy: u32 = 0u;
    var local_min_iz: u32 = 0u;

    let gid_flat = wid.x * 256u + tid;
    for (var idx = gid_flat; idx < n_total; idx = idx + 256u) {
        let val = values[idx];
        if val < local_min_val {
            local_min_val = val;
            let slice_size = nx * ny;
            let ixy = idx % slice_size;
            local_min_ix = ixy % nx;
            local_min_iy = ixy / nx;
            local_min_iz = idx / slice_size;
        }
    }

    shared_min_val[tid] = local_min_val;
    shared_min_ix[tid] = local_min_ix;
    shared_min_iy[tid] = local_min_iy;
    shared_min_iz[tid] = local_min_iz;
    workgroupBarrier();

    // Tree reduction within workgroup
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            let other_val = shared_min_val[tid + stride];
            if other_val < shared_min_val[tid] {
                shared_min_val[tid] = other_val;
                shared_min_ix[tid] = shared_min_ix[tid + stride];
                shared_min_iy[tid] = shared_min_iy[tid + stride];
                shared_min_iz[tid] = shared_min_iz[tid + stride];
            }
        }
        workgroupBarrier();
    }

    // Thread 0 writes workgroup minimum. Dispatch exactly 1 workgroup for
    // correct global minimum; for n_total > 256, use multiple workgroups and
    // a second reduction pass over partial results.
    if tid == 0u && wid.x == 0u {
        out_min_val[0u] = shared_min_val[0u];
        out_min_idx[0u] = shared_min_ix[0u];
        out_min_idx[1u] = shared_min_iy[0u];
        out_min_idx[2u] = shared_min_iz[0u];
    }
}
