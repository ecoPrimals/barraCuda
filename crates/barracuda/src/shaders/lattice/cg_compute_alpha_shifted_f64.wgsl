// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Shifted CG scalar: alpha = rz / (pAp + sigma * pp).
// Single-thread GPU kernel for shifted CG systems (D†D + σ)x = b.
// The shift σ is stored in a GPU buffer (set once per solve).
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

@group(0) @binding(0) var<storage, read> rz: array<f64>;
@group(0) @binding(1) var<storage, read> pap: array<f64>;
@group(0) @binding(2) var<storage, read> pp: array<f64>;
@group(0) @binding(3) var<storage, read> sigma: array<f64>;
@group(0) @binding(4) var<storage, read_write> alpha: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rz_val = rz[0];
    let pap_val = pap[0];
    let pp_val = pp[0];
    let sigma_val = sigma[0];
    let denom = pap_val + sigma_val * pp_val;
    if (abs(denom) > f64(1e-30)) {
        alpha[0] = rz_val / denom;
    } else {
        alpha[0] = f64(0.0);
    }
}
