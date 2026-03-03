// Fused seasonal pipeline: ET0 → Kc → Water Balance → Yield impact
// One GPU dispatch computes a full growing-season step per spatial cell.
//
// Provenance: airSpring V035 handoff → toadStool absorption (S70)
//
// REQUIRES: SHADER_F64 feature
// Architecture: One workgroup per spatial cell, cell data in flat arrays.

struct SeasonalParams {
    cell_count: u32,
    day_of_year: u32,
    stage_length: u32,      // crop development stage length (days)
    day_in_stage: u32,
    kc_prev: f64,           // Kc at stage start
    kc_next: f64,           // Kc at stage end
    taw_default: f64,
    raw_fraction: f64,
    field_capacity: f64,
    _pad0: u32,
    _pad1: u32,
}

// Per-cell inputs: [tmax, tmin, rh_max, rh_min, wind_2m, rs, elev, lat, soil_moisture_prev]
// 9 f64 per cell
@group(0) @binding(0) var<storage, read> cell_weather: array<f64>;
@group(0) @binding(1) var<storage, read_write> cell_output: array<f64>;
@group(0) @binding(2) var<uniform> params: SeasonalParams;

// FAO-56 Penman-Monteith ET₀ (inline, same as batched_elementwise_f64.wgsl)
fn et0_pm(tmax: f64, tmin: f64, rh_max: f64, rh_min: f64, wind: f64, rs: f64, elev: f64, lat: f64, doy: u32) -> f64 {
    let zero = tmax - tmax;
    let one = zero + 1.0;
    let pi = zero + 3.141592653589793;
    let tmean = (tmax + tmin) / (zero + 2.0);
    let p = (zero + 101.3) * pow_f64(((zero + 293.0) - (zero + 0.0065) * elev) / (zero + 293.0), zero + 5.26);
    let gamma = (zero + 0.000665) * p;
    let e_tx = (zero + 0.6108) * exp_f64((zero + 17.27) * tmax / (tmax + (zero + 237.3)));
    let e_tn = (zero + 0.6108) * exp_f64((zero + 17.27) * tmin / (tmin + (zero + 237.3)));
    let es = (e_tx + e_tn) / (zero + 2.0);
    let ea = (e_tn * rh_max / (zero + 100.0) + e_tx * rh_min / (zero + 100.0)) / (zero + 2.0);
    let e_tm = (zero + 0.6108) * exp_f64((zero + 17.27) * tmean / (tmean + (zero + 237.3)));
    let delta = (zero + 4098.0) * e_tm / pow_f64(tmean + (zero + 237.3), zero + 2.0);
    let lat_r = lat * pi / (zero + 180.0);
    let dr = one + (zero + 0.033) * cos_f64((zero + 2.0) * pi * f64(doy) / (zero + 365.0));
    let decl = (zero + 0.409) * sin_f64((zero + 2.0) * pi * f64(doy) / (zero + 365.0) - (zero + 1.39));
    var ws = acos_f64(-tan_f64(lat_r) * tan_f64(decl));
    if (ws != ws) { ws = pi; }
    let ra = (zero + 24.0) * (zero + 60.0) / pi * (zero + 0.082) * dr * (ws * sin_f64(lat_r) * sin_f64(decl) + cos_f64(lat_r) * cos_f64(decl) * sin_f64(ws));
    let rso = ((zero + 0.75) + (zero + 0.00002) * elev) * ra;
    let rns = (one - (zero + 0.23)) * rs;
    let sigma = zero + 0.000000004903;
    let rnl = sigma * (pow_f64(tmax + (zero + 273.16), zero + 4.0) + pow_f64(tmin + (zero + 273.16), zero + 4.0)) / (zero + 2.0) * ((zero + 0.34) - (zero + 0.14) * sqrt(ea)) * ((zero + 1.35) * rs / rso - (zero + 0.35));
    let rn = rns - rnl;
    let num = (zero + 0.408) * delta * rn + gamma * (zero + 900.0) / (tmean + (zero + 273.0)) * wind * (es - ea);
    let den = delta + gamma * (one + (zero + 0.34) * wind);
    return num / den;
}

@compute @workgroup_size(1)
fn seasonal_step(@builtin(workgroup_id) wg: vec3<u32>) {
    let cell = wg.x;
    if (cell >= params.cell_count) { return; }

    let zero = f64(0.0);
    let base = cell * 9u;
    let tmax = cell_weather[base + 0u];
    let tmin = cell_weather[base + 1u];
    let rh_max = cell_weather[base + 2u];
    let rh_min = cell_weather[base + 3u];
    let wind = cell_weather[base + 4u];
    let rs = cell_weather[base + 5u];
    let elev = cell_weather[base + 6u];
    let lat = cell_weather[base + 7u];
    let theta_prev = cell_weather[base + 8u];

    // Step 1: ET₀
    let et0 = et0_pm(tmax, tmin, rh_max, rh_min, wind, rs, elev, lat, params.day_of_year);

    // Step 2: Kc interpolation
    var kc = params.kc_prev;
    if (params.stage_length > 0u) {
        let frac = f64(params.day_in_stage) / f64(params.stage_length);
        kc = params.kc_prev + (params.kc_next - params.kc_prev) * frac;
    }

    // Step 3: ETc
    let etc = et0 * kc;

    // Step 4: Water balance
    let taw = params.taw_default;
    let raw = taw * params.raw_fraction;
    var ks = zero + 1.0;
    if (theta_prev > raw) {
        ks = max((taw - theta_prev) / max(taw - raw, zero + 0.001), zero);
    }
    let etc_adj = ks * etc;
    var theta_new = theta_prev - etc_adj;
    if (theta_new < zero) { theta_new = zero; }
    if (theta_new > params.field_capacity) { theta_new = params.field_capacity; }

    // Step 5: Stress indicator (yield impact proxy)
    let stress = zero + 1.0 - ks;

    // Output per cell: [et0, kc, etc, theta_new, stress]
    let out_base = cell * 5u;
    cell_output[out_base + 0u] = et0;
    cell_output[out_base + 1u] = kc;
    cell_output[out_base + 2u] = etc;
    cell_output[out_base + 3u] = theta_new;
    cell_output[out_base + 4u] = stress;
}
