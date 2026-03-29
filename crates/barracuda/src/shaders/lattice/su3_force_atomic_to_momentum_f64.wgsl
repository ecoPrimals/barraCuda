// SPDX-License-Identifier: AGPL-3.0-only
//
// Convert ROP atomic force accumulation (i32 fixed-point) back to f64
// and add to momentum buffer.
//
// Absorbed from hotSpring V0632 (March 2026).
//
// mom[i] += f64(force_accum[i]) / scale_factor
//
// Dispatched once after all fermion force poles have accumulated.

struct Params {
    n_values: u32,
    pad0: u32,
    inv_scale: f64,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> force_accum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> momentum: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if idx >= params.n_values { return; }

    let accum = atomicLoad(&force_accum[idx]);
    momentum[idx] += f64(accum) * params.inv_scale;
}
