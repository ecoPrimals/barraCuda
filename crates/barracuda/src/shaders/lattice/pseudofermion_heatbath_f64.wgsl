// SPDX-License-Identifier: AGPL-3.0-or-later
// pseudofermion_heatbath_f64.wgsl — Generate Gaussian noise for pseudofermion heatbath
//
// Prepend: complex_f64.wgsl + lcg_f64.wgsl
//
// Fills a fermion field η with Gaussian random ColorVectors (3 complex per site).
// The actual φ = D†η step is done via a separate Dirac operator dispatch.
//
// Buffer layout:
//   eta[V × 6]:       Gaussian noise field (3 colors × 2 for re/im = 6 f64/site)
//   rng_state[V]:     per-site RNG state

struct HeatbathParams {
    volume: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
}

@group(0) @binding(0) var<uniform>             params:    HeatbathParams;
@group(0) @binding(1) var<storage, read_write> eta:       array<f64>;
@group(0) @binding(2) var<storage, read_write> rng_state: array<u32>;

@compute @workgroup_size(64)
fn heatbath_noise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= params.volume) { return; }

    var state = rng_state[site];
    let base = site * 6u;

    // 3 color components, each complex (re, im)
    for (var c = 0u; c < 3u; c = c + 1u) {
        let re = prng_gaussian(&state);
        let im = prng_gaussian(&state);
        eta[base + c * 2u]      = re;
        eta[base + c * 2u + 1u] = im;
    }

    rng_state[site] = state;
}
