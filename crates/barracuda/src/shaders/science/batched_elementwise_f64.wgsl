// ============================================================================
// batched_elementwise_f64.wgsl — Unified batched element-wise computation
// ============================================================================
//
// UNIFIED PATTERN (Feb 16 2026) — Template for all springs:
//   - airSpring: Batched ET₀, water balance across stations/fields
//   - wetSpring: Batched diversity metrics across samples
//   - hotSpring: Batched nuclear structure across nuclei
//
// ARCHITECTURE:
//   - One workgroup per batch element
//   - Each workgroup processes one "row" of work
//   - Parameters passed via uniform buffer
//   - Input/output arrays are flattened [batch * stride]
//
// REQUIRES: SHADER_F64 feature
// Date: February 16, 2026
// License: AGPL-3.0-or-later
// ============================================================================

// ============================================================================
// OPERATION ENUM (selected via params.operation)
// ============================================================================
// 0 = FAO56_ET0 (Penman-Monteith reference evapotranspiration)
// 1 = WATER_BALANCE (daily depletion update)
// 2 = SHANNON_BATCH (Shannon entropy per sample)
// 3 = SIMPSON_BATCH (Simpson index per sample)
// 4 = CUSTOM (user-defined via auxiliary params)

// exp_f64, log_f64, pow_f64 provided by math_f64.wgsl auto-injection
// (full f64 precision via range-reduced minimax polynomials)

// ============================================================================
// FAO-56 PENMAN-MONTEITH (airSpring core equation)
// ============================================================================
// Input per station-day: [tmax, tmin, rh_max, rh_min, wind_2m, Rs, elevation, lat, doy]
// Output: ET₀ (mm/day)

fn fao56_et0(
    tmax: f64, tmin: f64,
    rh_max: f64, rh_min: f64,
    wind_2m: f64,
    rs: f64,
    elevation: f64,
    lat: f64,
    doy: u32
) -> f64 {
    let zero = tmax - tmax;
    let one = zero + 1.0;
    
    // Mean temperature
    let tmean = (tmax + tmin) / (zero + 2.0);
    
    // Atmospheric pressure (FAO-56 Eq. 7)
    let p = (zero + 101.3) * pow_f64((zero + 293.0) - (zero + 0.0065) * elevation, zero + 5.26) / pow_f64(zero + 293.0, zero + 5.26);
    
    // Psychrometric constant γ (kPa/°C)
    let gamma = (zero + 0.000665) * p;
    
    // Saturation vapour pressure
    let e_tmax = (zero + 0.6108) * exp_f64((zero + 17.27) * tmax / (tmax + (zero + 237.3)));
    let e_tmin = (zero + 0.6108) * exp_f64((zero + 17.27) * tmin / (tmin + (zero + 237.3)));
    let es = (e_tmax + e_tmin) / (zero + 2.0);
    
    // Actual vapour pressure (from RH)
    let ea = (e_tmin * rh_max / (zero + 100.0) + e_tmax * rh_min / (zero + 100.0)) / (zero + 2.0);
    
    // Slope of saturation vapour pressure curve Δ
    let e_tmean = (zero + 0.6108) * exp_f64((zero + 17.27) * tmean / (tmean + (zero + 237.3)));
    let delta = (zero + 4098.0) * e_tmean / pow_f64(tmean + (zero + 237.3), zero + 2.0);
    
    // Extraterrestrial radiation Ra (simplified)
    let pi = zero + 3.141592653589793;
    let lat_rad = lat * pi / (zero + 180.0);
    let dr = one + (zero + 0.033) * cos_f64((zero + 2.0) * pi * f64(doy) / (zero + 365.0));
    let decl = (zero + 0.409) * sin_f64((zero + 2.0) * pi * f64(doy) / (zero + 365.0) - (zero + 1.39));
    
    var ws = acos_f64(-tan_f64(lat_rad) * tan_f64(decl));
    if (ws != ws) { ws = pi; } // NaN check
    
    let gsc = zero + 0.0820;
    let ra = (zero + 24.0) * (zero + 60.0) / pi * gsc * dr * (
        ws * sin_f64(lat_rad) * sin_f64(decl) +
        cos_f64(lat_rad) * cos_f64(decl) * sin_f64(ws)
    );
    
    // Clear-sky radiation Rso
    let rso = ((zero + 0.75) + (zero + 0.00002) * elevation) * ra;
    
    // Net shortwave radiation
    let rns = (one - (zero + 0.23)) * rs;
    
    // Net longwave radiation (simplified Stefan-Boltzmann)
    let sigma = zero + 0.000000004903;  // MJ/(K⁴·m²·day)
    let tmax_k = tmax + (zero + 273.16);
    let tmin_k = tmin + (zero + 273.16);
    let rnl = sigma * (pow_f64(tmax_k, zero + 4.0) + pow_f64(tmin_k, zero + 4.0)) / (zero + 2.0) *
              ((zero + 0.34) - (zero + 0.14) * sqrt(ea)) *
              ((zero + 1.35) * rs / rso - (zero + 0.35));
    
    // Net radiation
    let rn = rns - rnl;
    
    // Soil heat flux (daily: G ≈ 0)
    let g = zero;
    
    // FAO-56 Penman-Monteith equation
    let numerator = (zero + 0.408) * delta * (rn - g) +
                    gamma * (zero + 900.0) / (tmean + (zero + 273.0)) * wind_2m * (es - ea);
    let denominator = delta + gamma * (one + (zero + 0.34) * wind_2m);
    
    return numerator / denominator;
}

// Precision trig for ET₀: sin_f64, cos_f64, tan_f64, acos_f64 from math_f64.wgsl
// (injected via compile_shader_f64 when shader calls them)

// ============================================================================
// BINDINGS
// ============================================================================

struct BatchParams {
    batch_size: u32,
    stride: u32,       // Elements per batch item
    operation: u32,    // Operation enum
    aux_param: f64,    // Auxiliary parameter (e.g., total for normalization)
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: BatchParams;

// ============================================================================
// MAIN: One workgroup per batch element
// ============================================================================
@compute @workgroup_size(64)
fn batched_compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }
    
    let base = batch_idx * params.stride;
    let op = params.operation;
    
    // Only thread 0 in each workgroup computes the result
    if (local_id.x != 0u) {
        return;
    }
    
    switch (op) {
        case 0u: {
            // FAO-56 ET₀: input is [tmax, tmin, rh_max, rh_min, wind, Rs, elev, lat, doy]
            let tmax = input[base + 0u];
            let tmin = input[base + 1u];
            let rh_max = input[base + 2u];
            let rh_min = input[base + 3u];
            let wind = input[base + 4u];
            let rs = input[base + 5u];
            let elev = input[base + 6u];
            let lat = input[base + 7u];
            let doy = u32(input[base + 8u]);
            
            output[batch_idx] = fao56_et0(tmax, tmin, rh_max, rh_min, wind, rs, elev, lat, doy);
        }
        case 1u: {
            // Water balance: simplified Dr update
            // input: [Dr_prev, P, I, ETc, TAW, RAW, p]
            let dr_prev = input[base + 0u];
            let precip = input[base + 1u];
            let irrig = input[base + 2u];
            let etc = input[base + 3u];
            let taw = input[base + 4u];
            let raw = input[base + 5u];
            let p_frac = input[base + 6u];
            
            let zero = dr_prev - dr_prev;
            
            // Stress coefficient Ks
            var ks = zero + 1.0;
            if (dr_prev > raw) {
                ks = (taw - dr_prev) / (taw - raw);
                if (ks < zero) { ks = zero; }
            }
            
            // Adjusted ETc
            let etc_adj = ks * etc;
            
            // New depletion
            var dr_new = dr_prev - precip - irrig + etc_adj;
            if (dr_new < zero) { dr_new = zero; }
            if (dr_new > taw) { dr_new = taw; }
            
            output[batch_idx] = dr_new;
        }
        case 5u: {
            // SensorCalibration: SoilWatch 10 VWC — Dong et al. (2024) Eq. 5
            // input: [raw_count]
            let raw = input[base + 0u];
            let c3 = raw * raw * raw;
            let c2 = raw * raw;
            output[batch_idx] = f64(2e-13) * c3 - f64(4e-9) * c2 + f64(4e-5) * raw - f64(0.0677);
        }
        case 6u: {
            // HargreavesEt0: Hargreaves-Samani (1985) — FAO-56 Eq. 52
            // input: [tmax, tmin, lat_rad, doy]
            let tmax = input[base + 0u];
            let tmin = input[base + 1u];
            let lat_rad = input[base + 2u];
            let doy_f = input[base + 3u];
            let zero = tmax - tmax;
            let two_pi = zero + 6.283185307179586;

            // Extraterrestrial radiation Ra (MJ/m²/day) — FAO-56 Eq. 21-25
            let dr = (zero + 1.0) + (zero + 0.033) * cos_f64(two_pi * doy_f / (zero + 365.0));
            let delta_decl = (zero + 0.409) * sin_f64(two_pi * doy_f / (zero + 365.0) - (zero + 1.39));
            var ws_arg = -tan_f64(lat_rad) * tan_f64(delta_decl);
            if (ws_arg > (zero + 1.0))  { ws_arg = zero + 1.0; }
            if (ws_arg < (zero - 1.0))  { ws_arg = zero - 1.0; }
            let ws = acos_f64(ws_arg);
            let ra_mj = (zero + 37.586) * dr * (
                ws * sin_f64(lat_rad) * sin_f64(delta_decl) +
                cos_f64(lat_rad) * cos_f64(delta_decl) * sin_f64(ws)
            );
            let ra_mm = ra_mj * (zero + 0.408);

            let tmean = (tmax + tmin) * (zero + 0.5);
            var td = tmax - tmin;
            if (td < zero) { td = zero; }
            var et0 = (zero + 0.0023) * (tmean + (zero + 17.8)) * sqrt(td) * ra_mm;
            if (et0 < zero) { et0 = zero; }
            output[batch_idx] = et0;
        }
        case 7u: {
            // KcClimateAdjust: FAO-56 Eq. 62
            // input: [kc_table, u2, rh_min, crop_height_m]
            let kc_table = input[base + 0u];
            let u2 = input[base + 1u];
            let rh_min = input[base + 2u];
            let h = input[base + 3u];
            let zero = kc_table - kc_table;

            let adj = ((zero + 0.04) * (u2 - (zero + 2.0)) - (zero + 0.004) * (rh_min - (zero + 45.0)))
                    * pow_f64(h / (zero + 3.0), zero + 0.3);
            var kc = kc_table + adj;
            if (kc < zero) { kc = zero; }
            output[batch_idx] = kc;
        }
        case 8u: {
            // DualKcKe: FAO-56 Eq. 71/74 soil evaporation coefficient
            // input: [kcb, kc_max, few, mulch_factor, de_prev, rew, tew, p_eff, et0]
            let kcb = input[base + 0u];
            let kc_max = input[base + 1u];
            let few = input[base + 2u];
            let mulch = input[base + 3u];
            let de_prev = input[base + 4u];
            let rew = input[base + 5u];
            let tew = input[base + 6u];
            let p_eff = input[base + 7u];
            // et0 = input[base + 8u]; reserved for future multi-output
            let zero = kcb - kcb;

            var de = de_prev - p_eff;
            if (de < zero) { de = zero; }
            if (de > tew)  { de = tew; }

            // Evaporation reduction coefficient Kr (FAO-56 Eq. 74)
            var kr = zero + 1.0;
            if (de > rew) {
                let denom = tew - rew;
                if (denom > (zero + 0.001)) {
                    kr = (tew - de) / denom;
                } else {
                    kr = zero;
                }
                if (kr < zero) { kr = zero; }
            }

            // Ke = min(Kr*(Kc_max - Kcb), few*Kc_max) * mulch (FAO-56 Eq. 71)
            let ke_full = kr * (kc_max - kcb);
            let ke_limit = few * kc_max;
            var ke = ke_full;
            if (ke > ke_limit) { ke = ke_limit; }
            ke = ke * mulch;
            if (ke < zero) { ke = zero; }
            output[batch_idx] = ke;
        }
        case 9u: {
            // VanGenuchtenTheta: θ(h) = θ_r + (θ_s - θ_r) / [1 + (α|h|)^n]^m, m = 1 - 1/n
            // input: [theta_r, theta_s, alpha, n, h]
            let theta_r = input[base + 0u];
            let theta_s = input[base + 1u];
            let alpha = input[base + 2u];
            let n_vg = input[base + 3u];
            let h = input[base + 4u];
            let zero = h - h;
            let one = zero + 1.0;

            if (h >= zero) {
                output[batch_idx] = theta_s;
            } else {
                let m = one - one / n_vg;
                let alpha_h = alpha * (-h);
                let alpha_h_n = pow_f64(alpha_h, n_vg);
                let one_plus = one + alpha_h_n;
                let se = pow_f64(one_plus, -m);
                output[batch_idx] = theta_r + (theta_s - theta_r) * se;
            }
        }
        case 10u: {
            // VanGenuchtenK: K(h) = K_s * S_e^l * [1 - (1 - S_e^(1/m))^m]^2
            // input: [K_s, theta_r, theta_s, alpha, n, l, h]
            let k_s = input[base + 0u];
            let theta_r = input[base + 1u];
            let theta_s = input[base + 2u];
            let alpha = input[base + 3u];
            let n_vg = input[base + 4u];
            let l_pore = input[base + 5u];
            let h = input[base + 6u];
            let zero = h - h;
            let one = zero + 1.0;

            if (h >= zero) {
                output[batch_idx] = k_s;
            } else {
                let m = one - one / n_vg;
                let alpha_h = alpha * (-h);
                let alpha_h_n = pow_f64(alpha_h, n_vg);
                let one_plus = one + alpha_h_n;
                let se = pow_f64(one_plus, -m);
                let se_1m = pow_f64(se, one / m);
                let inner = one - pow_f64(one - se_1m, m);
                output[batch_idx] = k_s * pow_f64(se, l_pore) * inner * inner;
            }
        }
        case 11u: {
            // ThornthwaiteEt0: ET₀ = 16 * (10*T_mean/I)^a * (N/12) * (d/30)
            // input: [heat_index_I, exponent_a, daylight_hours_N, days_in_month_d, T_mean]
            let heat_i = input[base + 0u];
            let exp_a = input[base + 1u];
            let daylight_n = input[base + 2u];
            let days_d = input[base + 3u];
            let t_mean = input[base + 4u];
            let zero = t_mean - t_mean;

            if (t_mean <= zero) {
                output[batch_idx] = zero;
            } else {
                let heat_safe = select(zero + 0.001, heat_i, heat_i > (zero + 0.001));
                output[batch_idx] = (zero + 16.0) * pow_f64((zero + 10.0) * t_mean / heat_safe, exp_a)
                    * (daylight_n / (zero + 12.0)) * (days_d / (zero + 30.0));
            }
        }
        case 12u: {
            // GDD: max(0, T_mean - T_base), aux_param = T_base
            let t_mean = input[base + 0u];
            let t_base = params.aux_param;
            var gdd = t_mean - t_base;
            if (gdd < (t_mean - t_mean)) { gdd = t_mean - t_mean; }
            output[batch_idx] = gdd;
        }
        case 13u: {
            // PedotransferPolynomial: Horner form a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*a5))))
            // input: [a0, a1, a2, a3, a4, a5, x]
            let a0 = input[base + 0u];
            let a1 = input[base + 1u];
            let a2 = input[base + 2u];
            let a3 = input[base + 3u];
            let a4 = input[base + 4u];
            let a5 = input[base + 5u];
            let x = input[base + 6u];
            output[batch_idx] = ((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0;
        }
        case 14u: {
            // MakkinkEt0: Makkink (1957) — radiation-based, Netherlands standard
            // ET₀ = 0.61 * (Δ/(Δ+γ)) * Rs/λ − 0.12
            // input: [Rs, T_mean, elevation]
            let rs_mj = input[base + 0u];
            let t_mean = input[base + 1u];
            let elev = input[base + 2u];
            let zero = t_mean - t_mean;

            // Atmospheric pressure P (kPa) — FAO-56 Eq. 7
            let p = (zero + 101.3) * pow_f64(((zero + 293.0) - (zero + 0.0065) * elev) / (zero + 293.0), zero + 5.26);
            let gamma = (zero + 0.000665) * p;

            // Δ — slope of saturation vapour pressure curve at T_mean
            let e_t = (zero + 0.6108) * exp_f64((zero + 17.27) * t_mean / (t_mean + (zero + 237.3)));
            let delta = (zero + 4098.0) * e_t / pow_f64(t_mean + (zero + 237.3), zero + 2.0);

            let lambda = zero + 2.45;  // MJ/kg (latent heat of vaporization)
            var et0 = (zero + 0.61) * (delta / (delta + gamma)) * rs_mj / lambda - (zero + 0.12);
            if (et0 < zero) { et0 = zero; }
            output[batch_idx] = et0;
        }
        case 15u: {
            // TurcEt0: Turc (1961) — radiation + temperature, humid climates
            // ET₀ = 0.013 * (T/(T+15)) * (Rs*23.8846 + 50) * C_rh
            // C_rh = 1.0 if RH ≥ 50%, else (1 + (50−RH)/70)
            // input: [Rs (MJ/m²/day), T_mean (°C), RH_mean (%)]
            let rs_mj = input[base + 0u];
            let t_mean = input[base + 1u];
            let rh = input[base + 2u];
            let zero = t_mean - t_mean;

            let rs_cal = rs_mj * (zero + 23.8846);  // MJ → cal/cm²/day
            var c_rh = zero + 1.0;
            if (rh < (zero + 50.0)) {
                c_rh = (zero + 1.0) + ((zero + 50.0) - rh) / (zero + 70.0);
            }
            var et0 = (zero + 0.013) * (t_mean / (t_mean + (zero + 15.0))) * (rs_cal + (zero + 50.0)) * c_rh;
            if (et0 < zero) { et0 = zero; }
            output[batch_idx] = et0;
        }
        case 16u: {
            // HamonEt0: Hamon (1963) — temperature + daylength only
            // PET (mm/day) = 13.97 × D² × Pt
            // where D = daylight / 12, Pt = 4.95 × exp(0.062 × T) / 100
            // (13.97 = 0.55 × 25.4 converts inches/day → mm/day)
            // input: [T_mean, daylight_hours]
            let t_mean = input[base + 0u];
            let daylight = input[base + 1u];
            let zero = t_mean - t_mean;

            let d_ratio = daylight / (zero + 12.0);
            let pt = (zero + 4.95) * exp_f64((zero + 0.062) * t_mean) / (zero + 100.0);
            var et0 = (zero + 13.97) * d_ratio * d_ratio * pt;
            if (et0 < zero) { et0 = zero; }
            output[batch_idx] = et0;
        }
        default: {
            // Identity / passthrough first element
            output[batch_idx] = input[base];
        }
    }
}
