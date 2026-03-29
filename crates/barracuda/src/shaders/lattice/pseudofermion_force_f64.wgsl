// SPDX-License-Identifier: AGPL-3.0-or-later
// pseudofermion_force_f64.wgsl — Per-link pseudofermion force for dynamical HMC
//
// Prepend: complex_f64.wgsl + su3.wgsl
//
// Computes dS_F/dU_μ(x) for all links from CG solution X and Y = D·X.
// The force is projected onto the su(3) algebra (traceless anti-Hermitian).
//
// Sign convention: the gauge force shader outputs ∂S_G/∂U (positive gradient)
// and momentum applies P += α_s·dt·F, so F = −η·TA[U(x⊗y†−y⊗x†)].
//
// Buffer layout:
//   links[V × 4 × 18]:    gauge links
//   x_field[V × 6]:       CG solution field (3 color components × 2 f64)
//   y_field[V × 6]:       D·X (Dirac applied to X)
//   force[V × 4 × 18]:    output force (algebra elements, accumulate)
//
// The staggered phase η_μ(x) = (-1)^{x_0+...+x_{μ-1}} is computed inline.

struct PFForceParams {
    nt:     u32,
    nx:     u32,
    ny:     u32,
    nz:     u32,
    volume: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
}

@group(0) @binding(0) var<uniform>             params:  PFForceParams;
@group(0) @binding(1) var<storage, read>       links:   array<f64>;
@group(0) @binding(2) var<storage, read>       x_field: array<f64>;
@group(0) @binding(3) var<storage, read>       y_field: array<f64>;
@group(0) @binding(4) var<storage, read_write> force:   array<f64>;

fn site_to_coords(s: u32) -> vec4<u32> {
    let nxyz = params.nx * params.ny * params.nz;
    let nyz  = params.ny * params.nz;
    let t    = s / nxyz;
    let rem  = s % nxyz;
    let x    = rem / nyz;
    let rem2 = rem % nyz;
    let y    = rem2 / params.nz;
    let z    = rem2 % params.nz;
    return vec4<u32>(t, x, y, z);
}

fn coords_to_site(c: vec4<u32>) -> u32 {
    return c.x * (params.nx * params.ny * params.nz)
         + c.y * (params.ny * params.nz)
         + c.z * params.nz
         + c.w;
}

fn shift_fwd(c: vec4<u32>, mu: u32) -> vec4<u32> {
    var r = c;
    switch (mu) {
        case 0u: { r.x = (c.x + 1u) % params.nt; }
        case 1u: { r.y = (c.y + 1u) % params.nx; }
        case 2u: { r.z = (c.z + 1u) % params.ny; }
        default: { r.w = (c.w + 1u) % params.nz; }
    }
    return r;
}

fn staggered_phase(c: vec4<u32>, mu: u32) -> f64 {
    var sum = 0u;
    for (var d = 0u; d < mu; d = d + 1u) {
        sum += c[d];
    }
    if ((sum & 1u) == 0u) { return f64(1.0); } else { return f64(-1.0); }
}

fn load_link(site: u32, mu: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let off = base + i * 2u;
        m[i] = c64_new(links[off], links[off + 1u]);
    }
    return m;
}

fn load_x_color(site: u32) -> array<vec2<f64>, 3> {
    var v: array<vec2<f64>, 3>;
    let base = site * 6u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        v[c] = c64_new(x_field[base + c * 2u], x_field[base + c * 2u + 1u]);
    }
    return v;
}

fn load_y_color(site: u32) -> array<vec2<f64>, 3> {
    var v: array<vec2<f64>, 3>;
    let base = site * 6u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        v[c] = c64_new(y_field[base + c * 2u], y_field[base + c * 2u + 1u]);
    }
    return v;
}

@compute @workgroup_size(64)
fn pseudofermion_force_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= params.volume) { return; }

    let coords = site_to_coords(site);

    for (var mu = 0u; mu < 4u; mu = mu + 1u) {
        let eta = staggered_phase(coords, mu);
        let fwd_coords = shift_fwd(coords, mu);
        let fwd_idx = coords_to_site(fwd_coords);
        let u = load_link(site, mu);

        var x_here = load_x_color(site);
        var x_fwd  = load_x_color(fwd_idx);
        var y_here = load_y_color(site);
        var y_fwd  = load_y_color(fwd_idx);

        // Build 3×3 outer product matrix M[a][b] with −η sign convention
        let neg_eta: f64 = f64(0.0) - eta;
        var m_mat: array<vec2<f64>, 9>;
        for (var a = 0u; a < 3u; a = a + 1u) {
            for (var b = 0u; b < 3u; b = b + 1u) {
                let c1 = c64_mul(x_fwd[a], c64_conj(y_here[b]));
                let c2 = c64_mul(y_fwd[a], c64_conj(x_here[b]));
                m_mat[a * 3u + b] = c64_scale(c64_sub(c1, c2), neg_eta);
            }
        }

        // W = U × M
        var w = su3_mul(u, m_mat);
        var wh = su3_adjoint(w);

        // Project onto algebra: ta = (W - W†)/2, then subtract trace/3
        var ta: array<vec2<f64>, 9>;
        for (var i = 0u; i < 9u; i = i + 1u) {
            ta[i] = c64_scale(c64_sub(w[i], wh[i]), f64(0.5));
        }
        let tr = c64_scale(c64_add(c64_add(ta[0], ta[4]), ta[8]), f64(1.0) / f64(3.0));
        ta[0] = c64_sub(ta[0], tr);
        ta[4] = c64_sub(ta[4], tr);
        ta[8] = c64_sub(ta[8], tr);

        // Store force
        let base = (site * 4u + mu) * 18u;
        var tav = ta;
        for (var i = 0u; i < 9u; i = i + 1u) {
            force[base + i * 2u]      = tav[i].x;
            force[base + i * 2u + 1u] = tav[i].y;
        }
    }
}
