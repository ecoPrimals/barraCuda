// SPDX-License-Identifier: AGPL-3.0-only
// higgs_u1_hmc_f64.wgsl — U(1) Abelian Higgs model HMC force computation
//
// Prepend: complex_f64.wgsl
//
// Computes the force update (leapfrog half-kick) for a 2D U(1) gauge field
// coupled to a complex Higgs scalar.  One kernel handles both link-angle and
// Higgs-field momenta in a single dispatch.
//
// Physics:
//   Action:   S = beta_pl * Σ_p (1 - cos θ_p)
//             + kappa * Σ_{x,mu} 2 * Re[ φ†(x) e^{i θ_mu(x)} φ(x+mu) ]
//             + lambda * Σ_x (|φ(x)|² - 1)²
//             + mu²   * Σ_x |φ(x)|²
//
//   Plaquette angle (2D): θ_p(x) = θ_0(x) + θ_1(x+0) - θ_0(x+1) - θ_1(x)
//   Gauge force:          F_θ(x,mu) = beta_pl * Σ_p∋mu Im(e^{i θ_p})
//                                    + 2*kappa * Im[ φ†(x) e^{i θ_mu(x)} φ(x+mu) ]
//   Higgs force (Wirtinger):
//     dS/dφ†(x) = kappa * Σ_mu [ e^{i θ_mu(x)} φ(x+mu) + e^{-i θ_mu(x-mu)} φ(x-mu) ]
//                + (2*lambda*(|φ|²-1) + mu²) * φ(x)
//
// CRITICAL: Factor of 2 in Wirtinger derivative.
//   dp_higgs/dt = -2 * dS/dφ†(x)    (not -dS/dφ†!)
//   Missing this factor causes |ΔH| >> 1 and ~0% HMC acceptance.
//
// Buffer layout (2D lattice, V = nt × ns):
//   link_angles[V × 2]:  θ_mu(x), 2 f64 per site (mu=0,1)
//   higgs      [V × 2]:  φ(x) = (re, im), 2 f64 per site
//   pi_links   [V × 2]:  conjugate momenta for link angles, updated in-place
//   pi_higgs   [V × 2]:  conjugate momenta for Higgs field (re,im), updated in-place
//
// hotSpring design: lattice/hmc.rs (v0.5.16, Feb 2026)
// Physics validated: |ΔH|/H < 1e-4 at dt=0.05 on 8×8 lattice.

struct HiggsParams {
    nt:      u32,
    ns:      u32,
    volume:  u32,
    _pad:    u32,
    beta_pl: f64,  // gauge plaquette coupling
    kappa:   f64,  // hopping parameter
    lambda:  f64,  // Higgs quartic coupling
    mu_sq:   f64,  // Higgs mass² parameter
    dt:      f64,  // leapfrog step size (half-kick = dt/2)
    _padf0:  f64,
    _padf1:  f64,
    _padf2:  f64,
}

@group(0) @binding(0) var<uniform>             params:    HiggsParams;
@group(0) @binding(1) var<storage, read>       links:     array<f64>; // [V×2] θ_mu
@group(0) @binding(2) var<storage, read>       higgs:     array<f64>; // [V×2] (re,im)
@group(0) @binding(3) var<storage, read_write> pi_links:  array<f64>; // [V×2] momentum
@group(0) @binding(4) var<storage, read_write> pi_higgs:  array<f64>; // [V×2] momentum

// ── Index helpers ─────────────────────────────────────────────────────────────

fn site(t: u32, x: u32) -> u32 { return t * params.ns + x; }

fn fwd(t: u32, x: u32, mu: u32) -> u32 {
    if (mu == 0u) { return site((t + 1u) % params.nt, x); }
    return site(t, (x + 1u) % params.ns);
}

fn bwd(t: u32, x: u32, mu: u32) -> u32 {
    if (mu == 0u) { return site((t + params.nt - 1u) % params.nt, x); }
    return site(t, (x + params.ns - 1u) % params.ns);
}

fn load_theta(s: u32, mu: u32) -> f64 { return links[s * 2u + mu]; }
fn load_phi(s: u32)  -> vec2<f64> { return c64_new(higgs[s * 2u], higgs[s * 2u + 1u]); }

// ── Plaquette angle θ_p(x) = θ_0(x) + θ_1(x+0) - θ_0(x+1) - θ_1(x) ─────────

fn plaquette_angle(t: u32, x: u32) -> f64 {
    let s    = site(t, x);
    let s_t  = fwd(t, x, 0u); // x + t-hat
    let s_x  = fwd(t, x, 1u); // x + x-hat
    return load_theta(s,   0u)
         + load_theta(s_t, 1u)
         - load_theta(s_x, 0u)
         - load_theta(s,   1u);
}

// ── Gauge force ───────────────────────────────────────────────────────────────
//
// F_θ_mu(x) = beta_pl * Im(e^{i θ_p}) summed over plaquettes touching link mu
//           + 2*kappa * Im[ φ†(x) * e^{i θ_mu(x)} * φ(x+mu) ]

fn gauge_force(t: u32, x: u32, mu: u32) -> f64 {
    let s    = site(t, x);
    let s_fw = fwd(t, x, mu);
    let nu   = 1u - mu; // the other direction in 2D

    // Sum over the two plaquettes touching link (s, mu).
    // Plaquette at x: contains (s, mu) with positive orientation
    var theta_p_fwd: f64 = 0.0;
    var theta_p_bwd: f64 = 0.0;
    if (mu == 0u) {
        // t-link at (t,x): plaquettes at (t,x) and (t-1,x)
        theta_p_fwd = plaquette_angle(t, x);
        let t_bwd = (t + params.nt - 1u) % params.nt;
        theta_p_bwd = -plaquette_angle(t_bwd, x); // appears with negative sign
    } else {
        // x-link at (t,x): plaquettes at (t,x) and (t,x-1)
        theta_p_fwd = -plaquette_angle(t, x);
        let x_bwd = (x + params.ns - 1u) % params.ns;
        theta_p_bwd = plaquette_angle(t, x_bwd);
    }
    // sin(f64) is not a WGSL builtin; extract Im(e^{iθ}) via c64_phase which
    // routes through the NVK exp/log workaround when compiled with compile_shader_f64.
    let gauge_f = params.beta_pl * (c64_phase(theta_p_fwd).y + c64_phase(theta_p_bwd).y);

    // Hopping term: Im[ φ†(x) e^{iθ} φ(x+mu) ]
    let phi_x  = load_phi(s);
    let phi_fw = load_phi(s_fw);
    let theta  = load_theta(s, mu);
    let hop    = c64_mul(c64_conj(phi_x), c64_mul(c64_phase(theta), phi_fw));
    let hop_f  = 2.0 * params.kappa * hop.y; // Im(...)

    return gauge_f + hop_f;
}

// ── Higgs force (Wirtinger) ───────────────────────────────────────────────────
//
// dS/dφ†(x) = kappa * Σ_mu [ e^{i θ_mu(x)} φ(x+mu) + e^{-i θ_mu(x-mu)} φ(x-mu) ]
//            + (2*lambda*(|φ|²-1) + mu²) * φ(x)
//
// Force: dp_higgs/dt = -2 * dS/dφ†(x)   ← factor 2 from Wirtinger convention

fn higgs_force(t: u32, x: u32) -> vec2<f64> {
    // Rewritten using raw f64 arithmetic to avoid Naga nested-call type
    // inference issues with c64_scale / c64_add on this wgpu version.
    let s = site(t, x);
    let phi = load_phi(s);
    let phi_abs_sq = phi.x * phi.x + phi.y * phi.y;

    // Covariant hopping sum: Σ_mu [ e^{iθ(x,mu)} φ(x+mu) + e^{-iθ(x-mu,mu)} φ(x-mu) ]
    var sum_re: f64 = 0.0;
    var sum_im: f64 = 0.0;
    for (var mu = 0u; mu < 2u; mu = mu + 1u) {
        // Forward hop: e^{i θ_mu(x)} φ(x+mu)
        let p_f   = c64_phase(load_theta(s, mu));
        let phi_f = load_phi(fwd(t, x, mu));
        sum_re = sum_re + p_f.x * phi_f.x - p_f.y * phi_f.y;
        sum_im = sum_im + p_f.x * phi_f.y + p_f.y * phi_f.x;

        // Backward hop: e^{-i θ_mu(x-mu)} φ(x-mu)
        let s_bw  = bwd(t, x, mu);
        let p_b   = c64_phase(-load_theta(s_bw, mu));
        let phi_b = load_phi(s_bw);
        sum_re = sum_re + p_b.x * phi_b.x - p_b.y * phi_b.y;
        sum_im = sum_im + p_b.x * phi_b.y + p_b.y * phi_b.x;
    }

    // dS/dφ† = kappa * covar_sum + (2λ(|φ|²-1) + μ²) * φ
    let potential = f64(2.0) * params.lambda * (phi_abs_sq - f64(1.0)) + params.mu_sq;
    let dS_re = params.kappa * sum_re + potential * phi.x;
    let dS_im = params.kappa * sum_im + potential * phi.y;

    // Wirtinger factor of -2:  dp/dt = -2 dS/dφ†
    return vec2<f64>(f64(-2.0) * dS_re, f64(-2.0) * dS_im);
}

// ── Kernel: half-kick momenta ─────────────────────────────────────────────────

@compute @workgroup_size(64)
fn hmc_half_kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.volume) { return; }

    let t = s / params.ns;
    let x = s % params.ns;
    let half_dt = params.dt * 0.5;

    // Update gauge momenta
    for (var mu = 0u; mu < 2u; mu = mu + 1u) {
        let f = gauge_force(t, x, mu);
        pi_links[s * 2u + mu] = pi_links[s * 2u + mu] - half_dt * f;
    }

    // Update Higgs momenta (Wirtinger force)
    let fh = higgs_force(t, x);
    pi_higgs[s * 2u]      = pi_higgs[s * 2u]      + half_dt * fh.x;
    pi_higgs[s * 2u + 1u] = pi_higgs[s * 2u + 1u] + half_dt * fh.y;
}
