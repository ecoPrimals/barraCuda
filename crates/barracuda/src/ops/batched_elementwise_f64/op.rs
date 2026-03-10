// SPDX-License-Identifier: AGPL-3.0-only
//! Operation types and input shapes for batched element-wise f64 compute.

/// Operations for batched element-wise computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Op {
    /// FAO-56 Penman-Monteith ET₀
    /// Input per batch: `[tmax, tmin, rh_max, rh_min, wind_2m, Rs, elevation, lat, doy]`
    Fao56Et0 = 0,

    /// Water balance daily update
    /// Input per batch: `[Dr_prev, P, I, ETc, TAW, RAW, p]`
    WaterBalance = 1,

    /// Custom operation (passthrough first element)
    Custom = 2,

    /// `SoilWatch` 10 sensor calibration — Dong et al. (2024)
    /// Input per batch: `[raw_count]`
    /// Output: VWC (cm³/cm³)
    SensorCalibration = 5,

    /// Hargreaves-Samani (1985) ET₀ — FAO-56 Eq. 52
    /// Input per batch: `[tmax, tmin, lat_rad, doy]`
    /// Output: ET₀ (mm/day)
    HargreavesEt0 = 6,

    /// FAO-56 Eq. 62 Kc climate adjustment
    /// Input per batch: `[kc_table, u2, rh_min, crop_height_m]`
    /// Output: adjusted Kc
    KcClimateAdjust = 7,

    /// FAO-56 Eq. 71/74 dual Kc soil evaporation coefficient
    /// Input per batch: `[kcb, kc_max, few, mulch_factor, de_prev, rew, tew, p_eff, et0]`
    /// Output: Ke
    DualKcKe = 8,

    /// Van Genuchten θ(h): soil water content from matric head
    /// Input per batch: `[theta_r, theta_s, alpha, n, h]`
    /// Output: volumetric water content θ (cm³/cm³)
    VanGenuchtenTheta = 9,

    /// Van Genuchten K(h): hydraulic conductivity from matric head
    /// Input per batch: `[K_s, theta_r, theta_s, alpha, n, l, h]`
    /// Output: hydraulic conductivity K
    VanGenuchtenK = 10,

    /// Thornthwaite monthly ET₀
    /// Input per batch: `[heat_index_I, exponent_a, daylight_hours_N, days_in_month_d, T_mean]`
    /// Output: ET₀ (mm/month)
    ThornthwaiteEt0 = 11,

    /// Growing Degree Days: max(0, `T_mean` - `T_base`)
    /// Input per batch: `[T_mean]` (user precomputes (`T_max+T_min)/2`)
    /// `aux_param`: `T_base` (base temperature)
    /// Output: GDD
    Gdd = 12,

    /// Pedotransfer polynomial: y = a0 + a1*x + a2*x² + ... (Horner, degree ≤5)
    /// Input per batch: `[a0, a1, a2, a3, a4, a5, x]`
    /// Output: polynomial evaluated at x
    PedotransferPolynomial = 13,

    /// Makkink (1957) ET₀ — radiation-based, Netherlands standard
    /// Input per batch: `[Rs, T_mean, elevation]`
    /// Output: ET₀ (mm/day)
    MakkinkEt0 = 14,

    /// Turc (1961) ET₀ — radiation + temperature, humid climates
    /// Input per batch: `[Rs, T_mean, RH_mean]`
    /// Output: ET₀ (mm/day)
    TurcEt0 = 15,

    /// Hamon (1963) ET₀ — temperature + daylength only
    /// Input per batch: `[T_mean, daylight_hours]`
    /// Output: ET₀ (mm/day)
    HamonEt0 = 16,

    /// SCS-CN (USDA TR-55) runoff estimation
    /// Input per batch: `[P, CN, Ia_ratio]`
    /// Output: Q (mm) runoff depth
    ScsCnRunoff = 17,

    /// Stewart (1977) yield-water function (Doorenbos & Kassam 1979)
    /// `Ya/Ym` = 1 - Ky × (1 - `ETa/ETc`)
    /// Input per batch: `[Ky, ETa_ETc_ratio]`
    /// Output: relative yield (`Ya/Ym`)
    StewartYieldWater = 18,

    /// Blaney-Criddle (1950) ET₀ — temperature + daylight percentage
    /// Input per batch: `[T_mean, p_daylight]`
    /// Output: ET₀ (mm/month)
    BlaneyCriddleEt0 = 19,
}

impl Op {
    /// Number of input elements per batch item
    #[must_use]
    pub fn stride(&self) -> usize {
        match self {
            Op::Fao56Et0 => 9,
            Op::WaterBalance => 7,
            Op::Custom => 1,
            Op::SensorCalibration => 1,
            Op::HargreavesEt0 => 4,
            Op::KcClimateAdjust => 4,
            Op::DualKcKe => 9,
            Op::VanGenuchtenTheta => 5,
            Op::VanGenuchtenK => 7,
            Op::ThornthwaiteEt0 => 5,
            Op::Gdd => 1,
            Op::PedotransferPolynomial => 7,
            Op::MakkinkEt0 => 3,
            Op::TurcEt0 => 3,
            Op::HamonEt0 => 2,
            Op::ScsCnRunoff => 3,
            Op::StewartYieldWater => 2,
            Op::BlaneyCriddleEt0 => 2,
        }
    }
}

/// FAO-56 station-day input: (tmax, tmin, `rh_max`, `rh_min`, `wind_2m`, rs, elevation, latitude, `day_of_year`)
pub type StationDayInput = (f64, f64, f64, f64, f64, f64, f64, f64, u32);

/// Water balance field input: (`dr_prev`, precipitation, irrigation, etc, taw, raw, `p_fraction`)
pub type WaterBalanceInput = (f64, f64, f64, f64, f64, f64, f64);
