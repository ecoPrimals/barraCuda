// SPDX-License-Identifier: AGPL-3.0-only
//
// groundSpring — Batched multinomial sampling for rarefaction

enable f64;
//
// Each invocation runs one replicate: draws `depth` reads from a
// community described by cumulative abundance probabilities, counting
// how many reads land in each taxon.
//
// Binding layout:
//   @group(0) @binding(0) params:      Params          {n_taxa, depth, n_reps, cumulative_probs, seed_lo, seed_hi, _pad}
//   @group(0) @binding(1) cumulative:  array<f64>      cumulative probabilities [n_taxa]
//   @group(0) @binding(2) seeds:       array<u32>      xoshiro state (4 × u32 per replicate)
//   @group(0) @binding(3) counts:      array<u32>      output [n_reps × n_taxa]
//
// Dispatch: (ceil(n_reps / 64), 1, 1)
//
// CPU reference: groundspring::rarefaction::multinomial_sample()
//
// For absorption into barracuda as ops::batched_multinomial_f64.
// Uses xoshiro128** matching barracuda::ops::prng_xoshiro_wgsl.

struct Params {
    n_taxa:          u32,
    depth:            u32,
    n_reps:           u32,
    cumulative_probs: u32,
    seed_lo:          u32,
    seed_hi:          u32,
    _pad0:            u32,
    _pad1:            u32,
}

@group(0) @binding(0) var<uniform>             params:     Params;
@group(0) @binding(1) var<storage, read>       cumulative: array<f64>;
@group(0) @binding(2) var<storage, read_write> seeds:      array<u32>;
@group(0) @binding(3) var<storage, read_write> counts:     array<u32>;

// ── Xoshiro128** (matches barracuda prng_xoshiro_wgsl) ──────────────

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn xoshiro_next(s: ptr<function, vec4<u32>>) -> u32 {
    let result = rotl((*s).y * 5u, 7u) * 9u;
    let t = (*s).y << 9u;
    (*s).z ^= (*s).x;
    (*s).w ^= (*s).y;
    (*s).y ^= (*s).z;
    (*s).x ^= (*s).w;
    (*s).z ^= t;
    (*s).w = rotl((*s).w, 11u);
    return result;
}

fn xoshiro_next_f64(s: ptr<function, vec4<u32>>) -> f64 {
    let hi = xoshiro_next(s);
    let lo = xoshiro_next(s);
    let combined = (f64(hi) * 4294967296.0 + f64(lo));
    return combined / 18446744073709551616.0;
}

// ── Binary search over cumulative probabilities ─────────────────────

fn find_taxon(u: f64) -> u32 {
    var lo = 0u;
    var hi = params.n_taxa;
    loop {
        if lo >= hi { break; }
        let mid = (lo + hi) / 2u;
        if cumulative[mid] < u {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return min(lo, params.n_taxa - 1u);
}

// ── Main ────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rep = gid.x;
    if rep >= params.n_reps { return; }

    let base = rep * params.n_taxa;
    let seed_base = rep * 4u;

    // Load per-replicate PRNG state
    var state = vec4<u32>(
        seeds[seed_base],
        seeds[seed_base + 1u],
        seeds[seed_base + 2u],
        seeds[seed_base + 3u],
    );

    // Zero output counts
    for (var t = 0u; t < params.n_taxa; t++) {
        counts[base + t] = 0u;
    }

    // Multinomial sampling: depth draws with binary-search assignment
    for (var d = 0u; d < params.depth; d++) {
        let u = xoshiro_next_f64(&state);
        let taxon = find_taxon(u);
        counts[base + taxon] += 1u;
    }

    // Write back PRNG state for potential multi-pass usage
    seeds[seed_base]      = state.x;
    seeds[seed_base + 1u] = state.y;
    seeds[seed_base + 2u] = state.z;
    seeds[seed_base + 3u] = state.w;
}
