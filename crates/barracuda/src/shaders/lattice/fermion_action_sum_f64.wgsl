// SPDX-License-Identifier: AGPL-3.0-or-later
//
// GPU-resident fermion action assembly for one RHMC sector:
//   S_f = alpha_0 * dots[0] + sum_s( alpha[s] * dots[s+1] )
//
// Accumulates into s_ferm_buf (adds to existing value so multiple sectors
// can call this sequentially). Caller must zero s_ferm_buf before H computation.
//
// Binding layout:
//   @binding(0) params:   array<f64>  [n_dots_f64, alpha_0]  (storage read)
//   @binding(1) dots:     array<f64>  (n_dots entries: phi·phi, phi·x_0, phi·x_1, ...)
//   @binding(2) alphas:   array<f64>  (n_dots-1 entries: alpha[0], alpha[1], ...)
//   @binding(3) s_ferm:   array<f64>  (1 entry, read_write — accumulated)
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

@group(0) @binding(0) var<storage, read> params: array<f64>;
@group(0) @binding(1) var<storage, read> dots: array<f64>;
@group(0) @binding(2) var<storage, read> alphas: array<f64>;
@group(0) @binding(3) var<storage, read_write> s_ferm: array<f64>;

@compute @workgroup_size(1)
fn main() {
    let n_dots = u32(params[0]);
    let alpha_0 = params[1];

    var action = alpha_0 * dots[0];
    for (var s = 1u; s < n_dots; s = s + 1u) {
        action = action + alphas[s - 1u] * dots[s];
    }

    s_ferm[0] = s_ferm[0] + action;
}
