// van_genuchten_f64.wgsl — Van Genuchten–Mualem soil hydraulic functions
//
// **Physics**: Soil water retention and hydraulic conductivity from pressure head h.
// Van Genuchten (1980): S_e(h) = [1 + (α|h|)^n]^(-m), m = 1 - 1/n
// Mualem (1976): K(S_e) = K_s * S_e^0.5 * [1 - (1 - S_e^(1/m))^m]²
//
// **Formulas**:
//   S_e(h) = [1 + (α|h|)^n]^(-m)   for h < 0; S_e = 1 for h >= 0
//   θ(h)   = θ_r + (θ_s - θ_r) * S_e   for h < 0; θ_s for h >= 0
//   K(h)   = K_s * S_e^0.5 * [1 - (1 - S_e^(1/m))^m]²
//   C(h)   = -α*m*n*(θ_s-θ_r) * (α|h|)^(n-1) * [1+(α|h|)^n]^(-m-1) for h < 0; 0 for h >= 0
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Params (f64 packed as u32 pairs: lo, hi): alpha, n_vg, m, theta_r, theta_s, k_s
//
// Reference: Van Genuchten (1980) SSSAJ; Mualem (1976) Water Resour Res

@group(0) @binding(0) var<storage, read> h_vals: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> S_e: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> theta: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> K: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> C: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

struct Params {
    n: u32,
    alpha: u32,
    alpha_hi: u32,
    n_vg: u32,
    n_vg_hi: u32,
    m: u32,
    m_hi: u32,
    theta_r: u32,
    theta_r_hi: u32,
    theta_s: u32,
    theta_s_hi: u32,
    k_s: u32,
    k_s_hi: u32,
}

fn unpack_f64(lo: u32, hi: u32) -> f64 {
    return bitcast<f64>(vec2<u32>(lo, hi));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    if (i >= n) {
        return;
    }

    let h = bitcast<f64>(h_vals[i]);

    // Unpack params
    let alpha = unpack_f64(params.alpha, params.alpha_hi);
    let n_vg = unpack_f64(params.n_vg, params.n_vg_hi);
    let m = unpack_f64(params.m, params.m_hi);
    let theta_r = unpack_f64(params.theta_r, params.theta_r_hi);
    let theta_s = unpack_f64(params.theta_s, params.theta_s_hi);
    let k_s = unpack_f64(params.k_s, params.k_s_hi);

    let zero = h - h;
    let one = zero + 1.0;

    if (h >= zero) {
        // Saturated: S_e = 1, θ = θ_s, K = K_s, C = 0
        S_e[i] = bitcast<vec2<u32>>(one);
        theta[i] = bitcast<vec2<u32>>(theta_s);
        K[i] = bitcast<vec2<u32>>(k_s);
        C[i] = vec2<u32>(0u, 0u);
        return;
    }

    let abs_h = -h;
    let alpha_h = alpha * abs_h;
    let alpha_h_n = pow(alpha_h, n_vg);
    let one_plus = one + alpha_h_n;
    let se = pow(one_plus, -m);

    // θ(h) = θ_r + (θ_s - θ_r) * S_e
    let theta_val = theta_r + (theta_s - theta_r) * se;
    theta[i] = bitcast<vec2<u32>>(theta_val);
    S_e[i] = bitcast<vec2<u32>>(se);

    // K(h) = K_s * S_e^0.5 * [1 - (1 - S_e^(1/m))^m]²
    let se_12 = sqrt(se);
    let se_1m = pow(se, one / m);
    let inner = one - pow(one - se_1m, m);
    let K_val = k_s * se_12 * inner * inner;
    K[i] = bitcast<vec2<u32>>(K_val);

    // C(h) = dθ/dh = (θ_s-θ_r)*dS_e/dh
    // dS_e/dh = α*m*n*(α|h|)^(n-1)*[1+(α|h|)^n]^(-m-1) for h<0
    // (Chain rule: S_e = (1+u^n)^(-m), u=α|h|, du/dh=-α for h<0; dS_e/dh = dS_e/du*du/dh)
    // Standard VG convention: C = (θ_s-θ_r)*α*m*n*(α|h|)^(n-1)*[1+(α|h|)^n]^(-m-1)
    let np1 = n_vg - one;
    let alpha_h_nm1 = select(one, pow(alpha_h, np1), n_vg > one);
    let one_plus_neg_m1 = pow(one_plus, -m - one);
    let dse_dh = alpha * m * n_vg * (theta_s - theta_r) * alpha_h_nm1 * one_plus_neg_m1;
    C[i] = bitcast<vec2<u32>>(dse_dh);
}
