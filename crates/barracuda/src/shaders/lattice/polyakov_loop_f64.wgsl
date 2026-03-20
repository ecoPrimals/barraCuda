// SPDX-License-Identifier: AGPL-3.0-or-later
// polyakov_loop_f64.wgsl — Polyakov loop (temporal Wilson line)
//
// Prepend: complex_f64.wgsl + su3.wgsl
//
// Computes Tr(Π_{t=0}^{Nt-1} U_3(t,x,y,z)) / 3  for each spatial site.
// Output: one complex value per spatial site (Re, Im).
//
// Buffer layout:
//   links[V × 4 × 18]:  gauge links
//   poly[V_spatial × 2]: Re/Im of Polyakov loop per spatial site

struct PolyParams {
    nt:           u32,
    nx:           u32,
    ny:           u32,
    nz:           u32,
    volume:       u32,
    spatial_vol:  u32,  // nx × ny × nz
    _pad0:        u32,
    _pad1:        u32,
}

@group(0) @binding(0) var<uniform>             params: PolyParams;
@group(0) @binding(1) var<storage, read>       links:  array<f64>;
@group(0) @binding(2) var<storage, read_write> poly:   array<f64>;

fn load_link(site: u32, mu: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let off = base + i * 2u;
        m[i] = c64_new(links[off], links[off + 1u]);
    }
    return m;
}

@compute @workgroup_size(64)
fn polyakov_loop_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spatial_idx = gid.x;
    if (spatial_idx >= params.spatial_vol) { return; }

    // Decompose spatial index into (ix, iy, iz)
    let nyz = params.ny * params.nz;
    let ix  = spatial_idx / nyz;
    let rem = spatial_idx % nyz;
    let iy  = rem / params.nz;
    let iz  = rem % params.nz;

    // Product of temporal links: Π_{t=0}^{Nt-1} U_3(t, ix, iy, iz)
    var prod = su3_identity();
    for (var t = 0u; t < params.nt; t = t + 1u) {
        let site = t * (params.nx * params.ny * params.nz)
                 + ix * (params.ny * params.nz)
                 + iy * params.nz
                 + iz;
        let u_t = load_link(site, 3u);
        prod = su3_mul(prod, u_t);
    }

    let tr = su3_trace(prod);
    let result = c64_scale(tr, f64(1.0) / f64(3.0));
    poly[spatial_idx * 2u]      = result.x;
    poly[spatial_idx * 2u + 1u] = result.y;
}
