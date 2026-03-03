// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU f64 moving window statistics.
//!
//! Provides f64-precision sliding window computations (mean, variance, min, max).
//! Complements the GPU f32 path in `ops::moving_window_stats`.
//!
//! # Provenance
//!
//! Absorbed from airSpring `metalForge/forge/src/moving_window_f64.rs` (V009).
//! Agricultural sensor data requires f64 for sub-degree temperature and
//! sub-percent soil moisture precision.

/// Result of a moving window statistics computation (f64 precision).
#[derive(Debug, Clone)]
pub struct MovingWindowResultF64 {
    /// Sliding window mean.
    pub mean: Vec<f64>,
    /// Sliding window variance (population).
    pub variance: Vec<f64>,
    /// Sliding window minimum.
    pub min: Vec<f64>,
    /// Sliding window maximum.
    pub max: Vec<f64>,
}

/// Compute moving window statistics over `data` with the given `window_size`.
///
/// Returns `None` if `data.len() < window_size` or `window_size == 0`.
/// Output vectors have length `data.len() - window_size + 1`.
#[must_use]
pub fn moving_window_stats_f64(data: &[f64], window_size: usize) -> Option<MovingWindowResultF64> {
    if data.len() < window_size || window_size == 0 {
        return None;
    }

    let out_len = data.len() - window_size + 1;
    let wf = window_size as f64;
    let mut mean = Vec::with_capacity(out_len);
    let mut variance = Vec::with_capacity(out_len);
    let mut min_vals = Vec::with_capacity(out_len);
    let mut max_vals = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let window = &data[i..i + window_size];
        let sum: f64 = window.iter().sum();
        let m = sum / wf;
        let var = window.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / wf;
        let wmin = window.iter().copied().fold(f64::INFINITY, f64::min);
        let wmax = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        mean.push(m);
        variance.push(var);
        min_vals.push(wmin);
        max_vals.push(wmax);
    }

    Some(MovingWindowResultF64 {
        mean,
        variance,
        min: min_vals,
        max: max_vals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_signal() {
        let data = vec![5.0; 100];
        let r = moving_window_stats_f64(&data, 10).unwrap();
        assert_eq!(r.mean.len(), 91);
        for &m in &r.mean {
            assert!((m - 5.0).abs() < 1e-12);
        }
        for &v in &r.variance {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_ramp() {
        let data: Vec<f64> = (0..20).map(f64::from).collect();
        let r = moving_window_stats_f64(&data, 5).unwrap();
        assert_eq!(r.mean.len(), 16);
        assert!((r.mean[0] - 2.0).abs() < 1e-12);
        assert!((r.mean[15] - 17.0).abs() < 1e-12);
        assert!(r.min[0].abs() < 1e-12);
        assert!((r.max[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_window_too_large() {
        assert!(moving_window_stats_f64(&[1.0, 2.0, 3.0], 5).is_none());
    }

    #[test]
    fn test_window_zero() {
        assert!(moving_window_stats_f64(&[1.0, 2.0], 0).is_none());
    }

    #[test]
    fn test_window_equals_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = moving_window_stats_f64(&data, 5).unwrap();
        assert_eq!(r.mean.len(), 1);
        assert!((r.mean[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_diurnal_smoothing() {
        use std::f64::consts::PI;
        let data: Vec<f64> = (0..168)
            .map(|i| {
                let hour = f64::from(i);
                8.0f64.mul_add(((hour % 24.0 - 14.0) * PI / 12.0).cos(), 25.0)
            })
            .collect();
        let r = moving_window_stats_f64(&data, 24).unwrap();
        assert_eq!(r.mean.len(), 145);
        for &m in &r.mean {
            assert!((m - 25.0).abs() < 0.5, "24h smoothed mean ≈ 25°C, got {m}");
        }
    }

    #[test]
    fn test_variance_known() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let r = moving_window_stats_f64(&data, 8).unwrap();
        assert_eq!(r.mean.len(), 1);
        assert!((r.mean[0] - 5.0).abs() < 1e-12);
        assert!((r.variance[0] - 4.0).abs() < 1e-12);
    }
}
