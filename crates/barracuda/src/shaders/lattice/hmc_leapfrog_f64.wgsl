// SPDX-License-Identifier: AGPL-3.0-or-later
// hmc_leapfrog_f64.wgsl — HMC leapfrog integration steps
//
// Prepend: complex_f64.wgsl + su3.wgsl + lcg_f64.wgsl + su3_extended_f64.wgsl
//
// Three entry points:
//   momentum_kick:   π ← π + dt × force    (half or full step)
//   link_update:     U ← exp(dt × π) × U   (Cayley + reunitarize)
//   generate_momenta: fill momenta with random su(3) algebra elements

struct LeapfrogParams {
    volume:  u32,
    n_links: u32,  // volume × 4
    _pad0:   u32,
    _pad1:   u32,
    dt:      f64,
    _padf:   f64,
}

@group(0) @binding(0) var<uniform>             params:    LeapfrogParams;
@group(0) @binding(1) var<storage, read_write> links:     array<f64>;
@group(0) @binding(2) var<storage, read_write> momenta:   array<f64>;
@group(0) @binding(3) var<storage, read>       force:     array<f64>;
@group(0) @binding(4) var<storage, read_write> rng_state: array<u32>;

fn load_link_rw(base: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        m[i] = c64_new(links[base + i * 2u], links[base + i * 2u + 1u]);
    }
    return m;
}

fn load_momentum(base: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        m[i] = c64_new(momenta[base + i * 2u], momenta[base + i * 2u + 1u]);
    }
    return m;
}

fn load_force(base: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        m[i] = c64_new(force[base + i * 2u], force[base + i * 2u + 1u]);
    }
    return m;
}

fn store_links(base: u32, m: array<vec2<f64>, 9>) {
    var mv = m;
    for (var i = 0u; i < 9u; i = i + 1u) {
        links[base + i * 2u]      = mv[i].x;
        links[base + i * 2u + 1u] = mv[i].y;
    }
}

fn store_momentum(base: u32, m: array<vec2<f64>, 9>) {
    var mv = m;
    for (var i = 0u; i < 9u; i = i + 1u) {
        momenta[base + i * 2u]      = mv[i].x;
        momenta[base + i * 2u + 1u] = mv[i].y;
    }
}

// π ← π + dt × force
@compute @workgroup_size(64)
fn momentum_kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let link_id = gid.x;
    if (link_id >= params.n_links) { return; }

    let base = link_id * 18u;
    let p = load_momentum(base);
    let f = load_force(base);
    let f_scaled = su3_scale(f, params.dt);
    let new_p = su3_add(p, f_scaled);
    store_momentum(base, new_p);
}

// U ← exp(dt × π) × U  then reunitarize
@compute @workgroup_size(64)
fn link_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let link_id = gid.x;
    if (link_id >= params.n_links) { return; }

    let base = link_id * 18u;
    let p = load_momentum(base);
    let u = load_link_rw(base);
    let exp_p = su3_exp_cayley(p, params.dt);
    let new_u = su3_reunitarize(su3_mul(exp_p, u));
    store_links(base, new_u);
}

// Generate random algebra momenta
@compute @workgroup_size(64)
fn generate_momenta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let link_id = gid.x;
    if (link_id >= params.n_links) { return; }

    var state = rng_state[link_id];
    let p = su3_random_algebra(&state);
    rng_state[link_id] = state;

    let base = link_id * 18u;
    store_momentum(base, p);
}
