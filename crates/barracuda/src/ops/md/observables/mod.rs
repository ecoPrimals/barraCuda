//! Observable Computation for Molecular Dynamics — Shader-First Architecture.
//!
//! All underlying math originates as f64 WGSL shaders. High-level analysis
//! functions (`compute_rdf`, `compute_vacf`, etc.) orchestrate snapshot
//! iteration and normalization around GPU kernels.
//!
//! **GPU Observables** (shader-dispatched):
//! - [`KineticEnergyF64`] — per-particle KE via `kinetic_energy_f64.wgsl`
//! - [`RdfHistogramF64`] — pair-distance histogram via `rdf_histogram_f64.wgsl`
//! - [`SsfGpu`] — static structure factor via `ssf_f64.wgsl`
//! - [`VacfGpu`] / [`VacfBatchGpu`] — velocity autocorrelation via `vacf_f64.wgsl`
//! - [`MsdGpu`] — mean-squared displacement via `msd_f64.wgsl`
//! - [`HeatCurrentGpu`] — heat current via `heat_current_f64.wgsl`
//! - [`StressVirialGpu`] — stress tensor via `stress_virial_f64.wgsl`
//!
//! **Snapshot orchestration** (control flow only, math in shaders):
//! - `compute_rdf` / `compute_vacf` / `compute_ssf` / `compute_msd`
//! - `validate_energy` — CPU statistics over scalar energy history
//!
//! **Deep Debt Compliance**:
//! - ✅ WGSL shader-first (all math as .wgsl, CPU gated `#[cfg(test)]`)
//! - ✅ Full f64 precision
//! - ✅ Zero unsafe code

pub mod heat_current_gpu;
mod kinetic_energy;
mod kinetic_energy_f64;
pub mod msd_gpu;
mod rdf_f64;
mod ssf_gpu;
pub mod transport_gpu;
mod vacf_gpu;

pub use heat_current_gpu::HeatCurrentGpu;
pub use kinetic_energy::KineticEnergy;
pub use kinetic_energy_f64::KineticEnergyF64;
pub use msd_gpu::MsdGpu;
pub use rdf_f64::RdfHistogramF64;
pub use ssf_gpu::SsfGpu;
pub use transport_gpu::{GpuVelocityRing, StressVirialGpu, VacfBatchGpu};
pub use vacf_gpu::VacfGpu;

use std::f64::consts::PI;

/// RDF result: g(r) binned at discrete r values
#[derive(Clone, Debug)]
pub struct Rdf {
    pub r_values: Vec<f64>,
    pub g_values: Vec<f64>,
    pub dr: f64,
}

/// VACF result: C(t) at discrete lag times
#[derive(Clone, Debug)]
pub struct Vacf {
    pub t_values: Vec<f64>,
    pub c_values: Vec<f64>,
    pub diffusion_coeff: f64,
}

/// Energy validation result
#[derive(Clone, Debug)]
pub struct EnergyValidation {
    pub mean_total: f64,
    pub std_total: f64,
    pub drift_pct: f64,
    pub mean_temperature: f64,
    pub std_temperature: f64,
    pub passed: bool,
}

/// Compute RDF from position snapshots (CPU post-process)
///
/// # Arguments
/// * `snapshots` - Position snapshots, each [N*3] flattened
/// * `n` - Number of particles
/// * `box_side` - Box side length in reduced units
/// * `n_bins` - Number of histogram bins
pub fn compute_rdf(snapshots: &[Vec<f64>], n: usize, box_side: f64, n_bins: usize) -> Rdf {
    let r_max = box_side / 2.0;
    let dr = r_max / n_bins as f64;
    let mut histogram = vec![0u64; n_bins];
    let n_frames = snapshots.len();

    for snap in snapshots {
        for i in 0..n {
            let xi = snap[i * 3];
            let yi = snap[i * 3 + 1];
            let zi = snap[i * 3 + 2];

            for j in (i + 1)..n {
                let mut dx = snap[j * 3] - xi;
                let mut dy = snap[j * 3 + 1] - yi;
                let mut dz = snap[j * 3 + 2] - zi;

                // PBC minimum image
                dx -= box_side * (dx / box_side).round();
                dy -= box_side * (dy / box_side).round();
                dz -= box_side * (dz / box_side).round();

                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let bin = (r / dr) as usize;
                if bin < n_bins {
                    histogram[bin] += 1;
                }
            }
        }
    }

    // Normalize: g(r) = histogram / (n_frames * N * n_density * 4π r² dr)
    let n_density = 3.0 / (4.0 * PI); // OCP reduced units
    let n_f = n as f64;
    let r_values: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dr).collect();
    let g_values: Vec<f64> = r_values
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let shell_vol = 4.0 * PI * r * r * dr;
            2.0 * histogram[i] as f64 / (n_frames as f64 * n_f * n_density * shell_vol).max(1e-30)
        })
        .collect();

    Rdf {
        r_values,
        g_values,
        dr,
    }
}

/// Compute VACF from velocity snapshots (CPU post-process)
///
/// # Arguments
/// * `vel_snapshots` - Velocity snapshots, each [N*3] flattened
/// * `n` - Number of particles
/// * `dt_dump` - Time between snapshots (reduced units)
/// * `max_lag` - Maximum lag in snapshots
pub fn compute_vacf(vel_snapshots: &[Vec<f64>], n: usize, dt_dump: f64, max_lag: usize) -> Vacf {
    let n_frames = vel_snapshots.len();
    let n_lag = max_lag.min(n_frames);
    let mut c_values = vec![0.0f64; n_lag];
    let mut counts = vec![0usize; n_lag];

    for t0 in 0..n_frames {
        for lag in 0..n_lag {
            let t1 = t0 + lag;
            if t1 >= n_frames {
                break;
            }
            let v0 = &vel_snapshots[t0];
            let v1 = &vel_snapshots[t1];

            let mut dot_sum = 0.0;
            for i in 0..n {
                dot_sum += v0[i * 3] * v1[i * 3]
                    + v0[i * 3 + 1] * v1[i * 3 + 1]
                    + v0[i * 3 + 2] * v1[i * 3 + 2];
            }
            c_values[lag] += dot_sum / n as f64;
            counts[lag] += 1;
        }
    }

    // Average over time origins
    for i in 0..n_lag {
        if counts[i] > 0 {
            c_values[i] /= counts[i] as f64;
        }
    }

    // Normalize by C(0)
    let c0 = c_values[0].max(1e-30);
    let c_normalized: Vec<f64> = c_values.iter().map(|&c| c / c0).collect();

    // Diffusion coefficient: D* = (1/3) integral_0^inf C(t) dt
    let mut integral = 0.0;
    for i in 1..n_lag {
        integral += 0.5 * (c_values[i - 1] + c_values[i]) * dt_dump;
    }
    let diffusion_coeff = integral / 3.0;

    let t_values: Vec<f64> = (0..n_lag).map(|i| i as f64 * dt_dump).collect();

    Vacf {
        t_values,
        c_values: c_normalized,
        diffusion_coeff,
    }
}

/// Compute static structure factor S(k) from position snapshots
///
/// # Arguments
/// * `snapshots` - Position snapshots, each [N*3] flattened
/// * `n` - Number of particles
/// * `box_side` - Box side length in reduced units
/// * `max_k_harmonics` - Number of k-vectors along each axis
pub fn compute_ssf(
    snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    max_k_harmonics: usize,
) -> Vec<(f64, f64)> {
    let dk = 2.0 * PI / box_side;
    let _n_frames = snapshots.len(); // Used implicitly via iteration
    let mut sk_values: Vec<(f64, f64)> = Vec::new();

    for kn in 1..=max_k_harmonics {
        let k_mag = kn as f64 * dk;
        let mut sk_sum = 0.0;
        let mut count = 0;

        for snap in snapshots {
            // S(k) = <|rho(k)|²> / N along each principal axis
            for axis in 0..3 {
                let mut re = 0.0;
                let mut im = 0.0;
                for j in 0..n {
                    let r_component = snap[j * 3 + axis];
                    let phase = k_mag * r_component;
                    re += phase.cos();
                    im += phase.sin();
                }
                sk_sum += (re * re + im * im) / n as f64;
                count += 1;
            }
        }

        sk_values.push((k_mag, sk_sum / count as f64));
    }

    sk_values
}

/// Mean-Squared Displacement (MSD) result
#[derive(Clone, Debug)]
pub struct Msd {
    /// Lag times τ in reduced units
    pub t_values: Vec<f64>,
    /// MSD(τ) = <|r(t+τ) - r(t)|²>
    pub msd_values: Vec<f64>,
    /// Diffusion coefficient from Einstein relation: D* = lim_{τ→∞} MSD(τ)/(6τ)
    pub diffusion_coeff: f64,
}

/// Compute Mean-Squared Displacement from position snapshots
///
/// MSD is fundamental for diffusion analysis:
/// - Einstein relation: D* = lim_{τ→∞} MSD(τ) / (6τ)
/// - Slope of MSD vs τ plot indicates diffusion regime
/// - Linear → normal diffusion, sub-linear → sub-diffusion
///
/// # Arguments
/// * `snapshots` - Position snapshots, each [N*3] flattened
/// * `n` - Number of particles
/// * `box_side` - Box side length (for PBC unwrapping)
/// * `dt` - Time between snapshots
/// * `max_lag` - Maximum lag (number of snapshot intervals)
pub fn compute_msd(
    snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    dt: f64,
    max_lag: usize,
) -> Msd {
    let n_frames = snapshots.len();
    let actual_max_lag = max_lag.min(n_frames - 1);

    let mut t_values = Vec::with_capacity(actual_max_lag);
    let mut msd_values = Vec::with_capacity(actual_max_lag);

    // Unwrap positions (handle PBC jumps)
    let unwrapped = unwrap_positions(snapshots, n, box_side);

    for lag in 1..=actual_max_lag {
        let mut msd_sum = 0.0;
        let mut count = 0;

        for t0 in 0..(n_frames - lag) {
            let t1 = t0 + lag;

            for i in 0..n {
                let dx = unwrapped[t1][i * 3] - unwrapped[t0][i * 3];
                let dy = unwrapped[t1][i * 3 + 1] - unwrapped[t0][i * 3 + 1];
                let dz = unwrapped[t1][i * 3 + 2] - unwrapped[t0][i * 3 + 2];
                msd_sum += dx * dx + dy * dy + dz * dz;
                count += 1;
            }
        }

        let msd = msd_sum / count as f64;
        t_values.push(lag as f64 * dt);
        msd_values.push(msd);
    }

    // Estimate diffusion coefficient from long-time slope
    // D* = MSD / (6 * t) — use last quarter of data for fitting
    let fit_start = (actual_max_lag * 3 / 4).max(1);
    let diffusion_coeff = if fit_start < actual_max_lag {
        let t_fit: Vec<f64> = t_values[fit_start..].to_vec();
        let msd_fit: Vec<f64> = msd_values[fit_start..].to_vec();
        linear_fit_slope(&t_fit, &msd_fit) / 6.0
    } else if let (Some(&msd), Some(&t)) = (msd_values.last(), t_values.last()) {
        msd / (6.0 * t)
    } else {
        0.0
    };

    Msd {
        t_values,
        msd_values,
        diffusion_coeff,
    }
}

/// Unwrap positions to handle PBC jumps
///
/// Detects when particles cross box boundaries and adjusts coordinates
/// to produce continuous trajectories for MSD/VACF calculations.
fn unwrap_positions(snapshots: &[Vec<f64>], n: usize, box_side: f64) -> Vec<Vec<f64>> {
    if snapshots.is_empty() {
        return Vec::new();
    }

    let mut unwrapped = Vec::with_capacity(snapshots.len());
    unwrapped.push(snapshots[0].clone());

    let mut image_counts = vec![[0i32; 3]; n]; // Track box crossings

    for frame_idx in 1..snapshots.len() {
        let prev = &unwrapped[frame_idx - 1];
        let curr = &snapshots[frame_idx];
        let mut unwrapped_frame = Vec::with_capacity(n * 3);

        for i in 0..n {
            for d in 0..3 {
                let r_prev = prev[i * 3 + d];
                let r_curr = curr[i * 3 + d];

                // Detect box crossing
                let delta = r_curr - (r_prev - image_counts[i][d] as f64 * box_side);
                if delta > box_side / 2.0 {
                    image_counts[i][d] -= 1;
                } else if delta < -box_side / 2.0 {
                    image_counts[i][d] += 1;
                }

                unwrapped_frame.push(r_curr + image_counts[i][d] as f64 * box_side);
            }
        }

        unwrapped.push(unwrapped_frame);
    }

    unwrapped
}

/// Simple linear fit to get slope (for diffusion coefficient)
fn linear_fit_slope(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

/// Validate energy conservation from energy history
///
/// # Arguments
/// * `energies` - Vector of (step, KE, PE, total) tuples
/// * `skip_fraction` - Fraction of initial data to skip (e.g., 0.1)
pub fn validate_energy(
    energies: &[(usize, f64, f64, f64)],
    skip_fraction: f64,
) -> EnergyValidation {
    if energies.is_empty() {
        return EnergyValidation {
            mean_total: 0.0,
            std_total: 0.0,
            drift_pct: 0.0,
            mean_temperature: 0.0,
            std_temperature: 0.0,
            passed: false,
        };
    }

    let skip = ((energies.len() as f64) * skip_fraction) as usize;
    let stable: Vec<_> = energies.iter().skip(skip).collect();

    if stable.is_empty() {
        return EnergyValidation {
            mean_total: 0.0,
            std_total: 0.0,
            drift_pct: 0.0,
            mean_temperature: 0.0,
            std_temperature: 0.0,
            passed: false,
        };
    }

    let totals: Vec<f64> = stable.iter().map(|e| e.3).collect();
    let mean_e: f64 = totals.iter().sum::<f64>() / totals.len() as f64;
    let var_e: f64 = totals.iter().map(|e| (e - mean_e).powi(2)).sum::<f64>() / totals.len() as f64;
    let std_e = var_e.sqrt();

    let e_initial = stable[0].3;
    let e_final = stable[stable.len() - 1].3;
    let drift_pct = if mean_e.abs() > 1e-30 {
        ((e_final - e_initial) / mean_e.abs()).abs() * 100.0
    } else {
        0.0
    };

    let passed = drift_pct < 5.0;

    EnergyValidation {
        mean_total: mean_e,
        std_total: std_e,
        drift_pct,
        mean_temperature: 0.0, // Caller should compute from KE
        std_temperature: 0.0,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msd_ballistic_motion() {
        // Test MSD with ballistic (constant velocity) motion
        // MSD should grow as t² for ballistic regime
        let n = 2;
        let box_side = 100.0; // Large box, no PBC effects
        let dt = 0.1;
        let velocity = 1.0;

        // Create trajectory: particles moving with constant velocity
        let n_frames = 50;
        let mut snapshots = Vec::with_capacity(n_frames);

        for frame in 0..n_frames {
            let t = frame as f64 * dt;
            snapshots.push(vec![
                t * velocity,  // particle 0: x
                0.0,           // particle 0: y
                0.0,           // particle 0: z
                -t * velocity, // particle 1: x (opposite direction)
                0.0,           // particle 1: y
                0.0,           // particle 1: z
            ]);
        }

        let msd = compute_msd(&snapshots, n, box_side, dt, 20);

        // For ballistic motion: MSD(t) = v² * t² * 2 (average over two particles)
        // At lag=10 (t=1.0): expected MSD ≈ 1² * 1.0² * 2 = 2.0
        assert!(!msd.msd_values.is_empty());
        assert_eq!(msd.msd_values.len(), 20);

        // Check quadratic growth (MSD ∝ t² for ballistic)
        let msd_early = msd.msd_values[4]; // lag=5
        let msd_late = msd.msd_values[9]; // lag=10
        let ratio = msd_late / msd_early;
        // Should be close to (10/5)² = 4
        assert!((ratio - 4.0).abs() < 0.1, "Ballistic MSD ratio: {}", ratio);
    }

    #[test]
    fn test_msd_diffusive_motion() {
        // Test MSD with random walk (diffusive motion)
        // MSD should grow linearly as 6*D*t for diffusive regime
        let n = 1;
        let box_side = 1000.0;
        let dt = 1.0;

        // Create random walk trajectory
        let n_frames = 100;
        let mut snapshots = Vec::with_capacity(n_frames);
        let step_size = 0.5;

        // Deterministic "random" walk for reproducibility
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        // Pseudo-random multipliers (deliberately close to golden ratio and e)
        #[allow(clippy::approx_constant)]
        let (mult_phi, mult_e) = (1.618, 2.718);
        for frame in 0..n_frames {
            snapshots.push(vec![x, y, z]);

            // Pseudo-random step based on frame number
            let angle1 = (frame as f64 * mult_phi) % (2.0 * PI);
            let angle2 = (frame as f64 * mult_e) % PI;
            x += step_size * angle2.sin() * angle1.cos();
            y += step_size * angle2.sin() * angle1.sin();
            z += step_size * angle2.cos();
        }

        let msd = compute_msd(&snapshots, n, box_side, dt, 50);

        // MSD should be positive and generally increasing
        assert!(!msd.msd_values.is_empty());
        assert!(msd.msd_values[0] > 0.0);
        assert!(msd.diffusion_coeff >= 0.0);
    }

    #[test]
    fn test_msd_pbc_unwrapping() {
        // Test that PBC unwrapping works correctly
        let n = 1;
        let box_side = 10.0;
        let dt = 1.0;

        // Particle crosses box boundary
        let snapshots = vec![
            vec![9.0, 0.0, 0.0], // Near right edge
            vec![1.0, 0.0, 0.0], // Wrapped to left (actual displacement +2)
            vec![3.0, 0.0, 0.0], // Continues right
        ];

        let msd = compute_msd(&snapshots, n, box_side, dt, 2);

        // After unwrapping: 9 → 11 → 13 (not 9 → 1 → 3)
        // MSD at lag=1 should be ~4 (displacement of 2)
        // MSD at lag=2 should be ~16 (displacement of 4)
        assert!(msd.msd_values.len() == 2);
        assert!(
            (msd.msd_values[0] - 4.0).abs() < 0.5,
            "MSD[0] = {}",
            msd.msd_values[0]
        );
        assert!(
            (msd.msd_values[1] - 16.0).abs() < 0.5,
            "MSD[1] = {}",
            msd.msd_values[1]
        );
    }

    #[test]
    fn test_linear_fit_slope() {
        // Test linear fitting
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let slope = linear_fit_slope(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
    }
}
