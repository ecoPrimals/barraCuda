// SPDX-License-Identifier: AGPL-3.0-only
// gillespie_ssa_f64.wgsl — Parallel Gillespie Stochastic Simulation Algorithm (f64)
//
// Each GPU thread runs one independent trajectory from t=0 to t_max using
// the direct method (Gillespie 1977).  All trajectories are fully parallel
// with no inter-thread communication.
//
// Algorithm per thread (one trajectory):
//   1. Compute propensities a_r = k_r × Π_s x_s^ν_r_s  (mass-action)
//   2. a0 = Σ a_r
//   3. If a0 ≤ 0 → system is absorbing, stop
//   4. τ  = -ln(u1) / a0           (waiting time, u1 ~ U(0,1))
//   5. Select reaction μ via linear search on cumulative propensities
//   6. Apply stoichiometry: x_s += net_stoich[μ, s]
//   7. Advance time: t += τ
//   8. Repeat until t ≥ t_max or max_steps exceeded
//
// PRNG: inline xoshiro128** (32-bit state per trajectory, 4 × u32)
//   State stored in prng_state[tid * 4 .. tid * 4 + 4].
//
// Propensities: stored in `propensities` storage buffer [T × R] — resolves
//   D-S21-002 (was limited to 32 reactions by function-scope static array).
//   Binding 7; Rust wrapper allocates T × R × 8 bytes.
//
// All f64 literals use f64() cast (naga rejects abstract 0.0/1.0 as f64).
//
// Absorbed from wetSpring handoff §P1 Gillespie (Feb 2026).

// ─── Parameters ──────────────────────────────────────────────────────────────

struct GillespieParams {
    n_reactions:    u32,   // number of reaction channels R
    n_species:      u32,   // number of chemical species S
    n_trajectories: u32,   // total number of parallel trajectories T
    max_steps:      u32,   // safety cap on iterations per trajectory
    t_max:          f64,   // simulation end time
    _pad0:          u32,
    _pad1:          u32,
}

// ─── Bindings ─────────────────────────────────────────────────────────────────

// Note: f64 not allowed in var<uniform>; pass as storage read buffer.
@group(0) @binding(0) var<storage, read>       params:       GillespieParams;
@group(0) @binding(1) var<storage, read>       rate_k:       array<f64>; // [R]
@group(0) @binding(2) var<storage, read>       stoich_react: array<u32>; // [R×S]
@group(0) @binding(3) var<storage, read>       stoich_net:   array<i32>; // [R×S]
@group(0) @binding(4) var<storage, read_write> states:       array<f64>; // [T×S]
@group(0) @binding(5) var<storage, read_write> prng_state:   array<u32>; // [T×4]
@group(0) @binding(6) var<storage, read_write> times:        array<f64>; // [T]
// Per-thread scratch propensity buffer — replaces static array<f64, 32>
// Supports arbitrary R; Rust allocates T × R × sizeof(f64) bytes.
@group(0) @binding(7) var<storage, read_write> propensities: array<f64>; // [T×R]

// ─── PRNG: xoshiro128** ───────────────────────────────────────────────────────

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn xoshiro_next(s: ptr<function, array<u32, 4>>) -> u32 {
    let result = rotl((*s)[1] * 5u, 7u) * 9u;
    let t = (*s)[1] << 9u;
    (*s)[2] ^= (*s)[0];
    (*s)[3] ^= (*s)[1];
    (*s)[1] ^= (*s)[2];
    (*s)[0] ^= (*s)[3];
    (*s)[2] ^= t;
    (*s)[3] = rotl((*s)[3], 11u);
    return result;
}

// Returns a uniform sample in (0, 1) as f64 using 32 bits of mantissa.
// Avoids exactly 0 (which would give log(0) = -∞) by adding 1 before dividing.
fn uniform01(s: ptr<function, array<u32, 4>>) -> f64 {
    let x = xoshiro_next(s);
    return (f64(x) + f64(1.0)) / f64(4294967297.0);  // (x + 1) / (2^32 + 1)
}

// ─── Main kernel ──────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= params.n_trajectories { return; }

    let R = params.n_reactions;
    let S = params.n_species;

    // Load PRNG state into registers
    var rng: array<u32, 4>;
    rng[0] = prng_state[tid * 4u + 0u];
    rng[1] = prng_state[tid * 4u + 1u];
    rng[2] = prng_state[tid * 4u + 2u];
    rng[3] = prng_state[tid * 4u + 3u];

    // Base offset into the per-thread propensity scratch area
    let prop_base = tid * R;

    var t: f64 = f64(0.0);

    for (var step = 0u; step < params.max_steps; step++) {
        // Step 1: Compute propensities → propensities[prop_base + r]
        var a0: f64 = f64(0.0);
        for (var r = 0u; r < R; r++) {
            var prop: f64 = rate_k[r];
            for (var s = 0u; s < S; s++) {
                let nu = stoich_react[r * S + s];
                var x_rem: f64 = states[tid * S + s];
                for (var cnt = 0u; cnt < nu; cnt++) {
                    if x_rem <= f64(0.0) {
                        prop = f64(0.0);
                    } else {
                        prop = prop * x_rem;
                    }
                    x_rem = x_rem - f64(1.0);
                }
            }
            if prop > f64(0.0) {
                propensities[prop_base + r] = prop;
            } else {
                propensities[prop_base + r] = f64(0.0);
            }
            a0 = a0 + propensities[prop_base + r];
        }

        // Step 2: Absorbing state check
        if a0 <= f64(0.0) { break; }

        // Step 3: Draw waiting time τ ~ Exp(a0)
        let u1 = uniform01(&rng);
        let tau = -log(u1) / a0;

        t = t + tau;
        if t >= params.t_max {
            t = params.t_max;
            break;
        }

        // Step 4: Select reaction μ via linear cumulative search
        let u2 = uniform01(&rng);
        let threshold: f64 = u2 * a0;
        var cumsum: f64 = f64(0.0);
        var mu = 0u;
        for (var r = 0u; r < R; r++) {
            cumsum = cumsum + propensities[prop_base + r];
            if cumsum >= threshold {
                mu = r;
                break;
            }
        }

        // Step 5: Apply net stoichiometry; clamp species counts at zero
        for (var s = 0u; s < S; s++) {
            let delta = stoich_net[mu * S + s];
            let new_val = states[tid * S + s] + f64(delta);
            if new_val > f64(0.0) {
                states[tid * S + s] = new_val;
            } else {
                states[tid * S + s] = f64(0.0);
            }
        }
    }

    // Write back PRNG state and final time
    prng_state[tid * 4u + 0u] = rng[0];
    prng_state[tid * 4u + 1u] = rng[1];
    prng_state[tid * 4u + 2u] = rng[2];
    prng_state[tid * 4u + 3u] = rng[3];
    times[tid] = t;
}
