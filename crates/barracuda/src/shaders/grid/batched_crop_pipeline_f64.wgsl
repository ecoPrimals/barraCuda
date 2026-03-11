// SPDX-License-Identifier: AGPL-3.0-only
//
// batched_crop_pipeline_f64.wgsl — Combined crop water balance
//
// soil_water[i+1] = soil_water[i] + precip[i] - ETc[i] - drainage[i]
// Simple water budget with drainage = max(0, soil_water - field_capacity).
//
// Layout: n_cells cells, each with n_steps time steps.
// precip, etc_vals: [n_cells * n_steps], index = cell * n_steps + step
// field_capacity: [n_cells]
// soil_water: [n_cells] (rw, in-place update)
// drainage: [n_cells * n_steps]
//
// Provenance: airSpring → ToadStool absorption

enable f64;

struct BatchedCropParams {
    n_steps: u32,
    n_cells: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> precip: array<f64>;
@group(0) @binding(1) var<storage, read> etc_vals: array<f64>;
@group(0) @binding(2) var<storage, read> field_capacity: array<f64>;
@group(0) @binding(3) var<storage, read_write> soil_water: array<f64>;
@group(0) @binding(4) var<storage, read_write> drainage: array<f64>;
@group(0) @binding(5) var<uniform> params: BatchedCropParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if cell >= params.n_cells { return; }

    let n_steps = params.n_steps;
    let fc = field_capacity[cell];
    var sw = soil_water[cell];

    for (var t = 0u; t < n_steps; t = t + 1u) {
        let idx = cell * n_steps + t;
        let drain = max(sw - fc, 0.0);
        sw = sw + precip[idx] - etc_vals[idx] - drain;
        sw = max(sw, 0.0);
        drainage[idx] = drain;
    }

    soil_water[cell] = sw;
}
