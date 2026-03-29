// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Multi-shift CG: ζ recurrence + shifted scalar computation.
// Jegerlehner (hep-lat/9612014) Algorithm 1.
//
// Per iteration j, for each shift s:
//   ζ_s^{j+1} = ζ_s^j * ζ_s^{j-1} * α_{j-1} /
//               (β_{j-1} * α_{j-1} * (ζ_s^{j-1} - ζ_s^j) +
//                ζ_s^{j-1} * α_{j-1} * (1 + σ_s * α_j))
//   α_s^j = α_j * ζ_s^{j+1} / ζ_s^j
//   ratio_s = ζ_s^{j+1} / ζ_s^j  (β_s = ratio² * β_base, computed in p-update)
//
// Then rotates: ζ_prev ← ζ_curr, ζ_curr ← ζ_new.
//
// All dynamic scalars (α_j, β_{j-1}, α_{j-1}) read from GPU buffers.
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

struct Params {
    n_shifts: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sigma: array<f64>;
@group(0) @binding(2) var<storage, read_write> zeta_curr: array<f64>;
@group(0) @binding(3) var<storage, read_write> zeta_prev: array<f64>;
@group(0) @binding(4) var<storage, read_write> alpha_s: array<f64>;
@group(0) @binding(5) var<storage, read_write> beta_ratio: array<f64>;
@group(0) @binding(6) var<storage, read> alpha_j: array<f64>;
@group(0) @binding(7) var<storage, read> beta_prev: array<f64>;
@group(0) @binding(8) var<storage, read> alpha_prev: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.n_shifts) { return; }

    let a_j = alpha_j[0];
    let b_prev = beta_prev[0];
    let a_prev = alpha_prev[0];

    let z = zeta_curr[s];
    let z_prev = zeta_prev[s];

    let numer = z * z_prev * a_prev;
    let denom = b_prev * a_prev * (z_prev - z)
              + z_prev * a_prev * (f64(1.0) + sigma[s] * a_j);

    var z_new = f64(0.0);
    if (abs(denom) > f64(1e-30)) {
        z_new = numer / denom;
    }

    var a_s = f64(0.0);
    if (abs(z) > f64(1e-30)) {
        a_s = a_j * z_new / z;
    }
    alpha_s[s] = a_s;

    var ratio = f64(0.0);
    if (abs(z) > f64(1e-30)) {
        ratio = z_new / z;
    }
    beta_ratio[s] = ratio;

    zeta_prev[s] = z;
    zeta_curr[s] = z_new;
}
