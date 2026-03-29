// SPDX-License-Identifier: AGPL-3.0-or-later
//
// GPU-resident Hamiltonian assembly: H = beta*(6V - plaq_sum) + T + S_f.
// Single-thread kernel. Eliminates CPU readback + assembly bottleneck.
//
// Binding layout:
//   @binding(0) params:    {beta: f64, six_v: f64}  (as array<f64, 2>, storage read)
//   @binding(1) plaq_sum:  f64  (GPU-resident scalar from reduce chain)
//   @binding(2) t:         f64  (GPU-resident kinetic energy from reduce chain)
//   @binding(3) s_ferm:    f64  (accumulated fermion action)
//   @binding(4) h_out:     f64  (output: assembled Hamiltonian)
//   @binding(5) diag_out:  {s_gauge: f64, t: f64, s_ferm: f64}  (diagnostics)
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

@group(0) @binding(0) var<storage, read> params: array<f64>;
@group(0) @binding(1) var<storage, read> plaq_sum: array<f64>;
@group(0) @binding(2) var<storage, read> t_ke: array<f64>;
@group(0) @binding(3) var<storage, read> s_ferm: array<f64>;
@group(0) @binding(4) var<storage, read_write> h_out: array<f64>;
@group(0) @binding(5) var<storage, read_write> diag_out: array<f64>;

@compute @workgroup_size(1)
fn main() {
    let beta = params[0];
    let six_v = params[1];

    let s_gauge = beta * (six_v - plaq_sum[0]);
    let t_val = t_ke[0];
    let sf_val = s_ferm[0];

    h_out[0] = s_gauge + t_val + sf_val;

    diag_out[0] = s_gauge;
    diag_out[1] = t_val;
    diag_out[2] = sf_val;
}
