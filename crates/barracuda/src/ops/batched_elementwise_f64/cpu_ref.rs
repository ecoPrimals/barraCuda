// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU reference implementations for batched element-wise ops.
//!
//! Used for tests and validation against GPU results.

/// Hargreaves-Samani ET₀ (CPU reference) — FAO-56 Eq. 52
pub fn hargreaves_et0_cpu(tmax: f64, tmin: f64, lat_rad: f64, doy: f64) -> f64 {
    use std::f64::consts::PI;
    let two_pi = 2.0 * PI;
    let dr = 1.0 + 0.033 * (two_pi * doy / 365.0).cos();
    let decl = 0.409 * (two_pi * doy / 365.0 - 1.39).sin();
    let ws_arg = (-lat_rad.tan() * decl.tan()).clamp(-1.0, 1.0);
    let ws = ws_arg.acos();
    let ra_mj =
        37.586 * dr * (ws * lat_rad.sin() * decl.sin() + lat_rad.cos() * decl.cos() * ws.sin());
    let ra_mm = ra_mj * 0.408;
    let tmean = (tmax + tmin) * 0.5;
    let td = (tmax - tmin).max(0.0);
    (0.0023 * (tmean + 17.8) * td.sqrt() * ra_mm).max(0.0)
}

/// FAO-56 Eq. 62 Kc climate adjustment (CPU reference)
pub fn kc_climate_adjust_cpu(kc_table: f64, u2: f64, rh_min: f64, crop_height_m: f64) -> f64 {
    let adj = (0.04 * (u2 - 2.0) - 0.004 * (rh_min - 45.0)) * (crop_height_m / 3.0_f64).powf(0.3);
    (kc_table + adj).max(0.0)
}

/// Van Genuchten θ(h) — soil water content from matric head (CPU reference)
/// θ(h) = θ_r + (θ_s - θ_r) / [1 + (α|h|)^n]^m where m = 1 - 1/n
pub fn van_genuchten_theta_cpu(theta_r: f64, theta_s: f64, alpha: f64, n: f64, h: f64) -> f64 {
    if h >= 0.0 {
        return theta_s;
    }
    let m = 1.0 - 1.0 / n;
    let alpha_h = alpha * (-h);
    let se = 1.0 / (1.0 + alpha_h.powf(n)).powf(m);
    theta_r + (theta_s - theta_r) * se
}

/// Van Genuchten K(h) — hydraulic conductivity (CPU reference)
/// K(h) = K_s * S_e^l * [1 - (1 - S_e^(1/m))^m]^2
pub fn van_genuchten_k_cpu(
    k_s: f64,
    _theta_r: f64,
    _theta_s: f64,
    alpha: f64,
    n: f64,
    l: f64,
    h: f64,
) -> f64 {
    if h >= 0.0 {
        return k_s;
    }
    let m = 1.0 - 1.0 / n;
    let alpha_h = alpha * (-h);
    let se = 1.0 / (1.0 + alpha_h.powf(n)).powf(m);
    let se_1m = se.powf(1.0 / m);
    let inner = 1.0 - (1.0 - se_1m).powf(m);
    k_s * se.powf(l) * inner * inner
}

/// Thornthwaite monthly ET₀ (CPU reference)
/// ET₀ = 16 * (10*T_mean/I)^a * (N/12) * (d/30)
pub fn thornthwaite_et0_cpu(
    heat_index_i: f64,
    exponent_a: f64,
    daylight_hours_n: f64,
    days_in_month_d: f64,
    t_mean: f64,
) -> f64 {
    if t_mean <= 0.0 {
        return 0.0;
    }
    16.0 * (10.0 * t_mean / heat_index_i.max(0.001)).powf(exponent_a)
        * (daylight_hours_n / 12.0)
        * (days_in_month_d / 30.0)
}

/// Pedotransfer polynomial — Horner form (CPU reference)
/// y = a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*a5))))
pub fn pedotransfer_polynomial_cpu(
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    x: f64,
) -> f64 {
    ((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0
}

#[cfg(test)]
/// FAO-56 Penman-Monteith ET₀ (CPU reference)
pub(crate) fn fao56_et0_cpu(
    tmax: f64,
    tmin: f64,
    rh_max: f64,
    rh_min: f64,
    wind_2m: f64,
    rs: f64,
    elevation: f64,
    lat: f64,
    doy: u32,
) -> f64 {
    use std::f64::consts::PI;

    let tmean = (tmax + tmin) / 2.0;

    // Atmospheric pressure (FAO-56 Eq. 7)
    let p = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0).powf(5.26);

    // Psychrometric constant
    let gamma = 0.000665 * p;

    // Saturation vapour pressure
    let e_tmax = 0.6108 * (17.27 * tmax / (tmax + 237.3)).exp();
    let e_tmin = 0.6108 * (17.27 * tmin / (tmin + 237.3)).exp();
    let es = (e_tmax + e_tmin) / 2.0;

    // Actual vapour pressure
    let ea = (e_tmin * rh_max / 100.0 + e_tmax * rh_min / 100.0) / 2.0;

    // Slope of saturation vapour pressure curve
    let e_tmean = 0.6108 * (17.27 * tmean / (tmean + 237.3)).exp();
    let delta = 4098.0 * e_tmean / (tmean + 237.3).powi(2);

    // Extraterrestrial radiation
    let lat_rad = lat * PI / 180.0;
    let dr = 1.0 + 0.033 * (2.0 * PI * doy as f64 / 365.0).cos();
    let decl = 0.409 * (2.0 * PI * doy as f64 / 365.0 - 1.39).sin();

    let tan_lat = lat_rad.tan();
    let tan_decl = decl.tan();
    let ws_arg = -tan_lat * tan_decl;
    let ws = if ws_arg.abs() > 1.0 {
        PI
    } else {
        ws_arg.acos()
    };

    let gsc = 0.0820;
    let ra = 24.0 * 60.0 / PI
        * gsc
        * dr
        * (ws * lat_rad.sin() * decl.sin() + lat_rad.cos() * decl.cos() * ws.sin());

    // Clear-sky radiation
    let rso = (0.75 + 0.00002 * elevation) * ra;

    // Net shortwave radiation
    let rns = (1.0 - 0.23) * rs;

    // Net longwave radiation
    let sigma = 4.903e-9; // Stefan-Boltzmann constant
    let tmax_k = tmax + 273.16;
    let tmin_k = tmin + 273.16;
    let rnl = sigma * (tmax_k.powi(4) + tmin_k.powi(4)) / 2.0
        * (0.34 - 0.14 * ea.sqrt())
        * (1.35 * rs / rso.max(0.001) - 0.35);

    // Net radiation
    let rn = rns - rnl;

    // FAO-56 Penman-Monteith equation
    let numerator = 0.408 * delta * rn + gamma * 900.0 / (tmean + 273.0) * wind_2m * (es - ea);
    let denominator = delta + gamma * (1.0 + 0.34 * wind_2m);

    numerator / denominator
}

#[cfg(test)]
/// Water balance daily update (CPU reference)
pub(crate) fn water_balance_cpu(
    dr_prev: f64,
    precip: f64,
    irrig: f64,
    etc: f64,
    taw: f64,
    raw: f64,
) -> f64 {
    // Stress coefficient
    let ks = if dr_prev > raw {
        ((taw - dr_prev) / (taw - raw)).max(0.0)
    } else {
        1.0
    };

    // Adjusted ETc
    let etc_adj = ks * etc;

    // New depletion
    let dr_new = dr_prev - precip - irrig + etc_adj;
    dr_new.clamp(0.0, taw)
}
