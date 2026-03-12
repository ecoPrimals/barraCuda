// SPDX-License-Identifier: AGPL-3.0-only
// Berendsen Thermostat (f64)
//
// **Physics**: Weak coupling to heat bath via velocity rescaling
// **Formula**: v *= sqrt(1 + (dt/τ) * (T_target/T_current - 1))
// **Use Case**: Equilibration phase — NOT for NVE production
//
// **Precision**: Full f64 (no math_f64 dependencies — pure arithmetic)
//
// The scale factor is computed on CPU from current temperature,
// then passed as a single parameter. The shader simply rescales.
//
// Bindings:
//   0: velocities [N*3] f64, read-write
//   1: params     [4]   f64, read  — [n, scale_factor, _, _]

@group(0) @binding(0) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let scale = params[1];

    velocities[i * 3u]      = velocities[i * 3u]      * scale;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] * scale;
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] * scale;
}
