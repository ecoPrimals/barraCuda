// SPDX-License-Identifier: AGPL-3.0-only
//
// van_genuchten_f64.wgsl — Van Genuchten soil moisture retention
//
// θ(h) = θ_r + (θ_s - θ_r) / [1 + (α|h|)^n]^m
// K(θ) = K_s * Se^l * [1 - (1 - Se^{1/m})^m]^2
// where Se = (θ - θ_r)/(θ_s - θ_r), m = 1 - 1/n
//
// Provenance: airSpring → ToadStool absorption

enable f64;

struct VanGenuchtenParams {
    n: u32,
    _pad: u32,
    theta_r: f64,
    theta_s: f64,
    alpha_vg: f64,
    n_vg: f64,
    k_s: f64,
    l_vg: f64,
}

@group(0) @binding(0) var<storage, read> h: array<f64>;
@group(0) @binding(1) var<storage, read_write> out_theta: array<f64>;
@group(0) @binding(2) var<storage, read_write> out_k: array<f64>;
@group(0) @binding(3) var<uniform> params: VanGenuchtenParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let theta_r = params.theta_r;
    let theta_s = params.theta_s;
    let alpha = params.alpha_vg;
    let n_vg = params.n_vg;
    let k_s = params.k_s;
    let l_vg = params.l_vg;

    let m = 1.0 - 1.0 / n_vg;
    let h_abs = abs(h[i]);
    let denom = 1.0 + pow(alpha * h_abs, n_vg);
    let theta = theta_r + (theta_s - theta_r) / pow(denom, m);
    out_theta[i] = theta;

    // K(θ): Se = (θ - θ_r)/(θ_s - θ_r), clamped to [0,1]
    let theta_range = theta_s - theta_r;
    let se = select(0.0, clamp((theta - theta_r) / theta_range, 0.0, 1.0), theta_range > 0.0);
    let se_1m = pow(se, 1.0 / m);
    let bracket = 1.0 - pow(1.0 - se_1m, m);
    out_k[i] = k_s * pow(se, l_vg) * bracket * bracket;
}
