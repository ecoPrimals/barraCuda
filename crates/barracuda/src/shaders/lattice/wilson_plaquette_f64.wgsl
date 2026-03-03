// wilson_plaquette_f64.wgsl — SU(3) Wilson plaquette action on a 4D lattice
//
// Prepend: complex_f64.wgsl + su3.wgsl
//
// Computes Re Tr(U_p) / 3 for all 6 plane orientations at every lattice site.
// Use ReduceScalarPipeline::sum_f64() on the output to get the average plaquette.
//
// Layout:
//   links[V × 4 × 18]:  link U_mu(x) at offset (site*4 + mu)*18
//   plaq [V × 6     ]:  Re Tr(U_p) / 3 for the 6 (mu<nu) planes
//
// Plane index ordering: (0,1)=tx (0,2)=ty (0,3)=tz (1,2)=xy (1,3)=xz (2,3)=yz
//
// Periodic boundary conditions are applied in all 4 directions.
//
// hotSpring design: lattice/wilson.rs (v0.5.16, Feb 2026)

struct PlaqParams {
    nt:     u32,
    nx:     u32,
    ny:     u32,
    nz:     u32,
    volume: u32,   // nt × nx × ny × nz
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
}

@group(0) @binding(0) var<uniform>             params: PlaqParams;
@group(0) @binding(1) var<storage, read>       links:  array<f64>; // [V × 4 × 18] f64
@group(0) @binding(2) var<storage, read_write> plaq:   array<f64>; // [V × 6] f64

// ── Coordinate helpers ────────────────────────────────────────────────────────

fn site_to_coords(s: u32) -> vec4<u32> {
    // Returns (t, x, y, z)
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

// Shift coordinate c in direction mu by +1 (periodic BC)
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

// Load link U_mu(site) from the flat links buffer
fn load_link(site: u32, mu: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let off = base + i * 2u;
        m[i] = c64_new(links[off], links[off + 1u]);
    }
    return m;
}

// ── Plaquette computation ─────────────────────────────────────────────────────

fn compute_plaquette(site: u32, mu: u32, nu: u32, coords: vec4<u32>) -> f64 {
    // U_p = U_mu(x) * U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)
    let site_fwd_mu = coords_to_site(shift_fwd(coords, mu));
    let site_fwd_nu = coords_to_site(shift_fwd(coords, nu));

    let u_mu       = load_link(site,       mu);
    let u_nu_fwd   = load_link(site_fwd_mu, nu);
    let u_mu_fwd   = load_link(site_fwd_nu, mu);
    let u_nu       = load_link(site,       nu);

    let plaq_mat = su3_plaquette(u_mu, u_nu_fwd, u_mu_fwd, u_nu);
    return su3_re_trace(plaq_mat) / 3.0;
}

// ── Kernel ────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn plaquette(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= params.volume) { return; }

    let coords = site_to_coords(site);

    // 6 plane orientations: (mu, nu) with mu < nu
    var plane_idx = 0u;
    for (var mu = 0u; mu < 4u; mu = mu + 1u) {
        for (var nu = mu + 1u; nu < 4u; nu = nu + 1u) {
            plaq[site * 6u + plane_idx] = compute_plaquette(site, mu, nu, coords);
            plane_idx = plane_idx + 1u;
        }
    }
}
