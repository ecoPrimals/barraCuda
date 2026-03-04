// SPDX-License-Identifier: AGPL-3.0-or-later
//
// groundSpring — Monte Carlo uncertainty propagation through FAO-56 ET₀
//
// STATUS: The FAO-56 equation chain itself is SUPERSEDED by
// barracuda::ops::BatchedElementwiseF64::fao56_et0_batch (Op::Fao56Et0).
// This shader remains as the MC uncertainty WRAPPER: it perturbs inputs
// with normal noise and dispatches through the equation chain.
//
// When absorbed, the compute_et0() call should be replaced with a call
// to the barracuda Fao56Et0 op, making this a thin MC driver.
//
// Binding layout:
//   @group(0) @binding(0) params:        Params         {n_samples, _pad}
//   @group(0) @binding(1) base_inputs:   array<f64, 9>  [tmax,tmin,rhmax,rhmin,wind,sun,lat,alt,doy]
//   @group(0) @binding(2) uncertainties: array<f64, 6>  [σ_tmax,σ_tmin,σ_rh,σ_rh,σ_wind_frac,σ_sun_frac]
//   @group(0) @binding(3) seeds:         array<u32>     xoshiro state (4 × u32 per sample)
//   @group(0) @binding(4) output:        array<f64>     one ET₀ per sample
//
// Dispatch: (ceil(n_samples / 64), 1, 1)
//
// CPU reference: validate_fao56::monte_carlo_et0()

struct Params {
    n_samples: u32,
    _pad:      u32,
    _pad2:     u32,
    _pad3:     u32,
}

@group(0) @binding(0) var<uniform>             params:        Params;
@group(0) @binding(1) var<storage, read>       base_inputs:   array<f64>;
@group(0) @binding(2) var<storage, read>       uncertainties: array<f64>;
@group(0) @binding(3) var<storage, read_write> seeds:         array<u32>;
@group(0) @binding(4) var<storage, read_write> output:        array<f64>;

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
    return (f64(hi) * 4294967296.0 + f64(lo)) / 18446744073709551616.0;
}

// ── Box-Muller normal variate ───────────────────────────────────────

const TAU: f64 = 6.28318530717958647692;

fn box_muller(s: ptr<function, vec4<u32>>) -> f64 {
    let u1 = max(xoshiro_next_f64(s), 5e-324);
    let u2 = xoshiro_next_f64(s);
    return sqrt(-2.0 * log(u1)) * cos(TAU * u2);
}

fn normal(s: ptr<function, vec4<u32>>, mu: f64, sigma: f64) -> f64 {
    return mu + sigma * box_muller(s);
}

// ── FAO-56 Penman-Monteith equation chain ───────────────────────────
//
// NOTE: When absorbed into barracuda, replace compute_et0() with a
// call to the existing Op::Fao56Et0 batched elementwise op.

const PI: f64  = 3.14159265358979323846;
const GSC: f64 = 0.0820;
const SIGMA_SB: f64 = 4.903e-9;
const ALBEDO: f64 = 0.23;

fn svp(t: f64) -> f64 {
    return 0.6108 * exp(17.27 * t / (t + 237.3));
}

fn compute_et0(tmax: f64, tmin: f64, rhmax: f64, rhmin: f64,
               wind_kmh: f64, sun_hrs: f64, lat: f64, alt: f64, doy: f64) -> f64 {
    let tmean = (tmax + tmin) / 2.0;
    let u2 = wind_kmh / 3.6 * 4.87 / log(67.8 * 10.0 - 5.42);
    let delta = 4098.0 * svp(tmean) / pow(tmean + 237.3, 2.0);
    let p = 101.3 * pow((293.0 - 0.0065 * alt) / 293.0, 5.26);
    let gamma = 0.000665 * p;
    let es = (svp(tmax) + svp(tmin)) / 2.0;
    let ea = (svp(tmin) * rhmax / 100.0 + svp(tmax) * rhmin / 100.0) / 2.0;
    let vpd = es - ea;

    let phi = lat * PI / 180.0;
    let dr = 1.0 + 0.033 * cos(2.0 * PI / 365.0 * doy);
    let decl = 0.409 * sin(2.0 * PI / 365.0 * doy - 1.39);
    let ws = acos(clamp(-tan(phi) * tan(decl), -1.0, 1.0));
    let ra = (24.0 * 60.0 / PI) * GSC * dr *
             (ws * sin(phi) * sin(decl) + cos(phi) * cos(decl) * sin(ws));
    let big_n = 24.0 / PI * ws;
    let n = clamp(sun_hrs, 0.0, big_n);
    let rs = (0.25 + 0.50 * n / big_n) * ra;
    let rso = (0.75 + 2e-5 * alt) * ra;
    let rns = (1.0 - ALBEDO) * rs;
    let rs_rso = select(0.7, min(rs / rso, 1.0), rso > 0.0);
    let rnl = SIGMA_SB * (pow(tmax + 273.16, 4.0) + pow(tmin + 273.16, 4.0)) / 2.0 *
              (0.34 - 0.14 * sqrt(ea)) * (1.35 * rs_rso - 0.35);
    let rn = rns - rnl;

    let num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd;
    let den = delta + gamma * (1.0 + 0.34 * u2);
    return num / den;
}

// ── Main compute shader ────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_samples { return; }

    let seed_base = idx * 4u;
    var state = vec4<u32>(
        seeds[seed_base],
        seeds[seed_base + 1u],
        seeds[seed_base + 2u],
        seeds[seed_base + 3u],
    );

    let tmax  = normal(&state, base_inputs[0], uncertainties[0]);
    let tmin  = min(normal(&state, base_inputs[1], uncertainties[1]), tmax - 1.0);
    let rhmax = clamp(normal(&state, base_inputs[2], uncertainties[2]), 10.0, 100.0);
    let rhmin = clamp(normal(&state, base_inputs[3], uncertainties[3]), 5.0, 100.0);
    let wind  = max(base_inputs[4] * (1.0 + uncertainties[4] * box_muller(&state)), 0.5);
    let sun   = max(base_inputs[5] * (1.0 + uncertainties[5] * box_muller(&state)), 0.0);

    output[idx] = max(compute_et0(tmax, tmin, rhmax, rhmin, wind, sun,
                              base_inputs[6], base_inputs[7], base_inputs[8]), 0.0);

    seeds[seed_base]      = state.x;
    seeds[seed_base + 1u] = state.y;
    seeds[seed_base + 2u] = state.z;
    seeds[seed_base + 3u] = state.w;
}
