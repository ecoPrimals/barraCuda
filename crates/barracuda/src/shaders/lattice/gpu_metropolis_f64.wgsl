// SPDX-License-Identifier: AGPL-3.0-or-later
//
// GPU-resident Metropolis accept/reject test (eliminates CPU readback bottleneck).
//
// Computes delta_H = H_new - H_old, uses the pre-generated uniform random
// from CPU (passed as param), and writes diagnostics. Single readback
// replaces all prior scalar readbacks for the entire trajectory.
//
// Binding layout:
//   @binding(0) params:     array<f64>  [rand, six_v]  (storage read)
//   @binding(1) h_old:      f64
//   @binding(2) h_new:      f64
//   @binding(3) plaq_sum:   f64  (from H_new's plaquette reduce — for reporting)
//   @binding(4) diag_old:   {s_gauge, t, s_ferm} as array<f64, 3>
//   @binding(5) diag_new:   {s_gauge, t, s_ferm} as array<f64, 3>
//   @binding(6) result:     array<f64>  (output, 9 entries)
//
// result layout:
//   [0] = accepted (1.0 or 0.0)
//   [1] = delta_h
//   [2] = plaquette (normalized: plaq_sum / 6V)
//   [3] = s_gauge_old
//   [4] = s_gauge_new
//   [5] = t_old
//   [6] = t_new
//   [7] = s_ferm_old
//   [8] = s_ferm_new
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

@group(0) @binding(0) var<storage, read> params: array<f64>;
@group(0) @binding(1) var<storage, read> h_old: array<f64>;
@group(0) @binding(2) var<storage, read> h_new: array<f64>;
@group(0) @binding(3) var<storage, read> plaq_sum: array<f64>;
@group(0) @binding(4) var<storage, read> diag_old: array<f64>;
@group(0) @binding(5) var<storage, read> diag_new: array<f64>;
@group(0) @binding(6) var<storage, read_write> result: array<f64>;

@compute @workgroup_size(1)
fn main() {
    let delta_h = h_new[0] - h_old[0];

    let r = params[0];
    let six_v = params[1];

    var accepted = f64(0.0);
    if (delta_h <= f64(0.0)) {
        accepted = f64(1.0);
    } else if (r < exp(-delta_h)) {
        accepted = f64(1.0);
    }

    result[0] = accepted;
    result[1] = delta_h;
    result[2] = plaq_sum[0] / six_v;
    result[3] = diag_old[0]; // s_gauge_old
    result[4] = diag_new[0]; // s_gauge_new
    result[5] = diag_old[1]; // t_old
    result[6] = diag_new[1]; // t_new
    result[7] = diag_old[2]; // s_ferm_old
    result[8] = diag_new[2]; // s_ferm_new
}
