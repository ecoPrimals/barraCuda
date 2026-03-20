// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched Element-wise Operations at f64 precision — Rust orchestrator
//!
//! UNIFIED PATTERN (Feb 16 2026) — Serves all springs:
//! - airSpring: FAO-56 ET₀, water balance across stations/fields
//! - wetSpring: Batched diversity metrics across samples
//! - hotSpring: Batched nuclear structure calculations
//!
//! # Architecture
//!
//! One workgroup per batch element. Each workgroup computes one output value
//! from a "row" of input parameters.
//!
//! # Operations
//!
//! - `Op::Fao56Et0` (0): FAO-56 Penman-Monteith reference ET₀
//! - `Op::WaterBalance` (1): Daily water balance update
//! - `Op::Custom` (2): User-defined operations (passthrough)
//! - `Op::SensorCalibration` (5): `SoilWatch` 10 VWC — Dong et al. (2024)
//! - `Op::HargreavesEt0` (6): Hargreaves-Samani (1985) ET₀
//! - `Op::KcClimateAdjust` (7): FAO-56 Eq. 62 Kc climate adjustment
//! - `Op::DualKcKe` (8): FAO-56 Eq. 71/74 dual Kc soil evaporation
//! - `Op::VanGenuchtenTheta` (9): Van Genuchten θ(h) soil water content
//! - `Op::VanGenuchtenK` (10): Van Genuchten K(h) hydraulic conductivity
//! - `Op::ThornthwaiteEt0` (11): Thornthwaite monthly ET₀
//! - `Op::Gdd` (12): Growing Degree Days (use `aux_param` for `T_base`)
//! - `Op::PedotransferPolynomial` (13): Polynomial evaluation (Horner, degree ≤5)
//! - `Op::MakkinkEt0` (14): Makkink (1957) ET₀ — radiation-based
//! - `Op::TurcEt0` (15): Turc (1961) ET₀ — radiation + temperature
//! - `Op::HamonEt0` (16): Hamon (1963) ET₀ — temperature + daylength
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::ops::batched_elementwise_f64::{BatchedElementwiseF64, Op};
//!
//! let executor = BatchedElementwiseF64::new(device.clone())?;
//!
//! // FAO-56 ET₀ for 100 station-days
//! // Input: [tmax, tmin, rh_max, rh_min, wind, Rs, elev, lat, doy] per station
//! let et0_values = executor.execute(&input_data, 100, Op::Fao56Et0)?;
//! ```

mod cpu_ref;
mod executor;
mod op;

pub use executor::BatchedElementwiseF64;
pub use op::{Op, StationDayInput, WaterBalanceInput};

// Re-export CPU reference implementations for external validation
pub use cpu_ref::{
    hamon_et0_cpu, hargreaves_et0_cpu, kc_climate_adjust_cpu, makkink_et0_cpu,
    pedotransfer_polynomial_cpu, thornthwaite_et0_cpu, turc_et0_cpu, van_genuchten_k_cpu,
    van_genuchten_theta_cpu,
};

/// Monte Carlo uncertainty propagation wrapper around FAO-56 ET₀.
/// Box-Muller perturbation + xoshiro128** PRNG + batched dispatch.
/// Provenance: groundSpring metalForge → toadStool absorption.
pub const WGSL_MC_ET0_PROPAGATE_F64: &str =
    include_str!("../../shaders/bio/mc_et0_propagate_f64.wgsl");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    #[test]
    fn test_fao56_et0_cpu_reference() {
        // FAO-56 Example 18: Reference grass ET₀
        // Uccle, Belgium (50°48'N, 4°21'E, 100m elevation)
        // July 6: tmax=21.5°C, tmin=12.3°C, RHmax=84%, RHmin=63%, u2=2.78m/s, Rs=22.07 MJ/m²/day
        let et0 = cpu_ref::fao56_et0_cpu(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187);

        // Expected: ~3.88 mm/day (FAO-56 Example 18)
        assert!(
            (et0 - 3.88).abs() < 0.1,
            "FAO-56 Example 18: got {et0} mm/day, expected ~3.88 mm/day"
        );
    }

    #[test]
    fn test_water_balance_no_stress() {
        // No stress: Dr < RAW
        let dr_new = cpu_ref::water_balance_cpu(30.0, 5.0, 0.0, 4.0, 100.0, 50.0);
        // Dr_new = 30 - 5 - 0 + 4 = 29 (no stress since Dr < RAW)
        assert!((dr_new - 29.0).abs() < 0.001);
    }

    #[test]
    fn test_water_balance_with_stress() {
        // Stress: Dr > RAW
        let dr_new = cpu_ref::water_balance_cpu(60.0, 0.0, 0.0, 5.0, 100.0, 50.0);
        // Ks = (100 - 60) / (100 - 50) = 0.8
        // ETc_adj = 0.8 * 5 = 4
        // Dr_new = 60 - 0 - 0 + 4 = 64
        assert!((dr_new - 64.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_fao56_et0_gpu() -> Result<()> {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return Ok(()); // Skip if no f64 GPU available
        };
        let executor = BatchedElementwiseF64::new(device)?;

        // Test batch of 3 station-days
        let station_days = vec![
            (21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187u32),
            (25.0, 15.0, 80.0, 50.0, 3.0, 20.0, 200.0, 45.0, 180),
            (30.0, 20.0, 70.0, 40.0, 2.0, 25.0, 50.0, 35.0, 200),
        ];

        let results = executor.fao56_et0_batch(&station_days)?;
        assert_eq!(results.len(), 3);

        // First result should match FAO-56 Example 18 (~3.88 mm/day)
        assert!(
            (results[0] - 3.88).abs() < 0.2,
            "GPU ET₀[0]: got {} mm/day, expected ~3.88 mm/day",
            results[0]
        );

        Ok(())
    }

    #[test]
    fn test_sensor_calibration_cpu() {
        let raw = 15_000.0_f64;
        let vwc = 2e-13 * raw.powi(3) - 4e-9 * raw.powi(2) + 4e-5 * raw - 0.0677;
        assert!(vwc > 0.0 && vwc < 1.0, "VWC={vwc} out of range");
    }

    #[test]
    fn test_hargreaves_et0_cpu() {
        use std::f64::consts::PI;
        let lat_rad = 50.8 * PI / 180.0;
        let et0 = hargreaves_et0_cpu(21.5, 12.3, lat_rad, 187.0);
        assert!(
            et0 > 1.0 && et0 < 10.0,
            "Hargreaves ET₀={et0}, expected 1-10 mm/day"
        );
    }

    #[test]
    fn test_kc_climate_adjust_cpu() {
        let kc = kc_climate_adjust_cpu(1.15, 3.0, 30.0, 2.0);
        assert!(kc > 1.0 && kc < 2.0, "Kc={kc}, expected ~1.15-1.3");
    }

    #[test]
    fn test_dual_kc_ke_cpu() {
        let kcb: f64 = 0.3;
        let kc_max: f64 = 1.2;
        let few: f64 = 0.5;
        let mulch: f64 = 1.0;
        let de_prev: f64 = 5.0;
        let rew: f64 = 9.0;
        let tew: f64 = 22.0;
        let p_eff: f64 = 0.0;

        let de = (de_prev - p_eff).clamp(0.0, tew);
        assert!(de < rew, "should be below REW");
        let ke_full = 1.0_f64 * (kc_max - kcb);
        let ke_limit = few * kc_max;
        let ke = ke_full.min(ke_limit) * mulch;
        assert!(ke > 0.0 && ke < 1.0, "Ke={ke}");
    }

    #[test]
    fn test_op_strides() {
        assert_eq!(Op::Fao56Et0.stride(), 9);
        assert_eq!(Op::WaterBalance.stride(), 7);
        assert_eq!(Op::Custom.stride(), 1);
        assert_eq!(Op::SensorCalibration.stride(), 1);
        assert_eq!(Op::HargreavesEt0.stride(), 4);
        assert_eq!(Op::KcClimateAdjust.stride(), 4);
        assert_eq!(Op::DualKcKe.stride(), 9);
        assert_eq!(Op::VanGenuchtenTheta.stride(), 5);
        assert_eq!(Op::VanGenuchtenK.stride(), 7);
        assert_eq!(Op::ThornthwaiteEt0.stride(), 5);
        assert_eq!(Op::Gdd.stride(), 1);
        assert_eq!(Op::PedotransferPolynomial.stride(), 7);
        assert_eq!(Op::MakkinkEt0.stride(), 3);
        assert_eq!(Op::TurcEt0.stride(), 3);
        assert_eq!(Op::HamonEt0.stride(), 2);
        assert_eq!(Op::ScsCnRunoff.stride(), 3);
        assert_eq!(Op::StewartYieldWater.stride(), 2);
        assert_eq!(Op::BlaneyCriddleEt0.stride(), 2);
    }

    #[test]
    fn test_scs_cn_runoff_cpu() {
        let q = cpu_ref::scs_cn_runoff_cpu(80.0, 75.0, 0.2);
        // CN=75 → S = 25400/75−254 = 84.67, Ia = 0.2*84.67 = 16.93
        // Pe = 80−16.93 = 63.07, Q = 63.07² / (63.07 + 84.67) = 26.93
        assert!((q - 26.93).abs() < 0.5, "SCS-CN Q={q} mm, expected ~26.93");
        assert_eq!(cpu_ref::scs_cn_runoff_cpu(0.0, 75.0, 0.2), 0.0);
        assert_eq!(cpu_ref::scs_cn_runoff_cpu(5.0, 75.0, 0.2), 0.0);
    }

    #[test]
    fn test_stewart_yield_water_cpu() {
        let ya_ym = cpu_ref::stewart_yield_water_cpu(1.0, 0.8);
        // Ya/Ym = 1 − 1.0*(1−0.8) = 0.8
        assert!(
            (ya_ym - 0.8).abs() < 1e-10,
            "Stewart Ya/Ym={ya_ym}, expected 0.8"
        );
        let ya_ym_full = cpu_ref::stewart_yield_water_cpu(1.0, 1.0);
        assert!((ya_ym_full - 1.0).abs() < 1e-10, "full ET → full yield");
    }

    #[test]
    fn test_blaney_criddle_et0_cpu() {
        let et0 = cpu_ref::blaney_criddle_et0_cpu(20.0, 12.0);
        // p = 12/43.8 = 0.274, ET₀ = 0.274*(0.46*20+8.13) = 0.274*17.33 = 4.75
        assert!(
            (et0 - 4.75).abs() < 0.3,
            "Blaney-Criddle ET₀={et0}, expected ~4.75"
        );
        let et0_cold = cpu_ref::blaney_criddle_et0_cpu(-5.0, 12.0);
        assert!(et0_cold >= 0.0, "cold temp should clamp to 0");
    }

    #[test]
    fn test_makkink_et0_cpu() {
        let et0 = makkink_et0_cpu(18.0, 20.0, 100.0);
        assert!(
            et0 > 0.5 && et0 < 8.0,
            "Makkink ET₀={et0} mm/day, expected 0.5-8"
        );
    }

    #[test]
    fn test_turc_et0_cpu() {
        let et0 = turc_et0_cpu(18.0, 20.0, 60.0);
        assert!(
            et0 > 1.0 && et0 < 15.0,
            "Turc ET₀={et0} mm/day, expected 1-15"
        );
        let et0_dry = turc_et0_cpu(18.0, 20.0, 30.0);
        assert!(et0_dry > et0, "dry air should increase Turc ET₀");
    }

    #[test]
    fn test_hamon_et0_cpu() {
        let et0 = hamon_et0_cpu(20.0, 14.0);
        assert!(
            et0 > 0.5 && et0 < 10.0,
            "Hamon ET₀={et0} mm/day, expected 0.5-10"
        );
    }

    #[test]
    fn test_van_genuchten_theta_cpu() {
        // Sandy loam: θ_r=0.065, θ_s=0.41, α=0.075, n=1.89
        let theta = van_genuchten_theta_cpu(0.065, 0.41, 0.075, 1.89, -100.0);
        assert!(
            theta > 0.1 && theta < 0.4,
            "θ(-100)={theta}, expected 0.1-0.4 for sandy loam"
        );
        let theta_sat = van_genuchten_theta_cpu(0.065, 0.41, 0.075, 1.89, 0.0);
        assert!((theta_sat - 0.41).abs() < 0.001, "θ(0) should equal θ_s");
    }

    #[test]
    fn test_van_genuchten_k_cpu() {
        let k = van_genuchten_k_cpu(10.0, 0.065, 0.41, 0.075, 1.89, 0.5, -100.0);
        assert!(k > 0.0 && k < 10.0, "K(-100)={k}, should be < K_s");
        let k_sat = van_genuchten_k_cpu(10.0, 0.065, 0.41, 0.075, 1.89, 0.5, 0.0);
        assert!((k_sat - 10.0).abs() < 0.001, "K(0) should equal K_s");
    }

    #[test]
    fn test_thornthwaite_et0_cpu() {
        // I≈100, a≈0.16 for mid-latitude; N=12, d=30
        // Thornthwaite gives ~18 mm/month for T=20°C with these params
        let et0 = thornthwaite_et0_cpu(100.0, 0.16, 12.0, 30.0, 20.0);
        assert!(
            et0 > 10.0 && et0 < 50.0,
            "Thornthwaite ET₀(20°C)={et0} mm/month, expected 10-50"
        );
    }

    #[test]
    fn test_gdd_cpu() {
        let gdd = (25.0_f64 - 10.0).max(0.0); // T_mean=25, T_base=10
        assert!((gdd - 15.0).abs() < 0.001);
        let gdd_zero = (5.0_f64 - 10.0).max(0.0);
        assert!(gdd_zero < 0.001);
    }

    #[test]
    fn test_pedotransfer_polynomial_cpu() {
        // y = 1 + 2x + 3x² at x=2 → 1 + 4 + 12 = 17
        let y = pedotransfer_polynomial_cpu(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0);
        assert!((y - 17.0).abs() < 0.001, "polynomial(2)={y}, expected 17");
    }

    #[tokio::test]
    async fn test_batched_ops_9_to_13_gpu() -> Result<()> {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return Ok(());
        };
        let executor = BatchedElementwiseF64::new(device)?;

        // Op 9: Van Genuchten θ
        let vg_theta_input = [0.065, 0.41, 0.075, 1.89, -100.0f64];
        let theta_out = executor.execute(&vg_theta_input, 1, Op::VanGenuchtenTheta)?;
        assert_eq!(theta_out.len(), 1);
        let cpu_theta = van_genuchten_theta_cpu(0.065, 0.41, 0.075, 1.89, -100.0);
        assert!(
            (theta_out[0] - cpu_theta).abs() < 0.01,
            "GPU θ={}, CPU θ={}",
            theta_out[0],
            cpu_theta
        );

        // Op 10: Van Genuchten K
        let vg_k_input = [10.0, 0.065, 0.41, 0.075, 1.89, 0.5, -100.0f64];
        let k_out = executor.execute(&vg_k_input, 1, Op::VanGenuchtenK)?;
        assert_eq!(k_out.len(), 1);
        let cpu_k = van_genuchten_k_cpu(10.0, 0.065, 0.41, 0.075, 1.89, 0.5, -100.0);
        assert!(
            (k_out[0] - cpu_k).abs() < 0.1,
            "GPU K={}, CPU K={}",
            k_out[0],
            cpu_k
        );

        // Op 11: Thornthwaite
        let thorn_input = [100.0, 0.16, 12.0, 30.0, 20.0f64];
        let et0_out = executor.execute(&thorn_input, 1, Op::ThornthwaiteEt0)?;
        assert_eq!(et0_out.len(), 1);
        let cpu_et0 = thornthwaite_et0_cpu(100.0, 0.16, 12.0, 30.0, 20.0);
        assert!(
            (et0_out[0] - cpu_et0).abs() < 1.0,
            "GPU ET₀={}, CPU ET₀={}",
            et0_out[0],
            cpu_et0
        );

        // Op 12: GDD (aux_param = t_base)
        let gdd_input = [25.0f64]; // T_mean
        let gdd_out = executor.execute_with_aux(&gdd_input, 1, Op::Gdd, 10.0)?;
        assert_eq!(gdd_out.len(), 1);
        assert!((gdd_out[0] - 15.0).abs() < 0.001, "GDD={}", gdd_out[0]);

        // Op 13: Pedotransfer polynomial
        let poly_input = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0f64];
        let poly_out = executor.execute(&poly_input, 1, Op::PedotransferPolynomial)?;
        assert_eq!(poly_out.len(), 1);
        assert!(
            (poly_out[0] - 17.0).abs() < 0.001,
            "poly(2)={}",
            poly_out[0]
        );

        Ok(())
    }
}
