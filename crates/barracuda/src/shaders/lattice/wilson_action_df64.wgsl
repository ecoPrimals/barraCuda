// wilson_action_df64.wgsl — Hybrid per-site Wilson action (DF64 core streaming)
//
// Prepend: complex_f64.wgsl + su3.wgsl + df64_core.wgsl + su3_df64.wgsl
//
// HYBRID PRECISION STRATEGY:
//   DF64 (FP32 cores): plaquette products (4 SU(3) matmuls × 6 planes/site)
//   f64  (FP64 units): 1 - Re Tr(P)/3 per-site accumulation
//
// Buffer layout: UNCHANGED from wilson_action_f64.wgsl.
//   links[V × 4 × 18]: f64 gauge links
//   action[V]:          f64 per-site action contribution
//
// Full Wilson action = β × Σ_x action[x]  (reduction on host).

struct ActionParams {
    nt:     u32,
    nx:     u32,
    ny:     u32,
    nz:     u32,
    volume: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
}

@group(0) @binding(0) var<uniform>             params: ActionParams;
@group(0) @binding(1) var<storage, read>       links:  array<f64>;
@group(0) @binding(2) var<storage, read_write> action: array<f64>;

// ── Coordinate helpers ───────────────────────────────────────────────────────

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

// ── DF64 link loading ────────────────────────────────────────────────────────

fn load_link_df64(site: u32, mu: u32) -> array<Cdf64, 9> {
    var m: array<Cdf64, 9>;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let off = base + i * 2u;
        m[i] = cdf64_from_f64(links[off], links[off + 1u]);
    }
    return m;
}

// ── Kernel ───────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn wilson_action_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= params.volume) { return; }

    let coords = site_to_coords(site);
    var s: f64 = 0.0;

    for (var mu = 0u; mu < 4u; mu = mu + 1u) {
        for (var nu = mu + 1u; nu < 4u; nu = nu + 1u) {
            let site_fwd_mu = coords_to_site(shift_fwd(coords, mu));
            let site_fwd_nu = coords_to_site(shift_fwd(coords, nu));

            // DF64 zone: plaquette product on FP32 cores
            let u_mu     = load_link_df64(site, mu);
            let u_nu_fwd = load_link_df64(site_fwd_mu, nu);
            let u_mu_fwd = load_link_df64(site_fwd_nu, mu);
            let u_nu     = load_link_df64(site, nu);

            let plaq_mat = su3_plaquette_df64(u_mu, u_nu_fwd, u_mu_fwd, u_nu);

            // Boundary: DF64 → f64 for trace and accumulation
            let re_tr = df64_to_f64(su3_re_trace_df64(plaq_mat));
            s += f64(1.0) - re_tr / f64(3.0);
        }
    }

    action[site] = s;
}
