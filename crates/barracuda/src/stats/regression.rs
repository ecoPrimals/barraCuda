// SPDX-License-Identifier: AGPL-3.0-or-later
//! Analytical least-squares regression primitives.
//!
//! Pure-Rust closed-form regression models (no iterative optimization).
//!
//! | Model | Equation | Method | Min points |
//! |-------|----------|--------|:----------:|
//! | [`fit_linear`] | y = a·x + b | Normal equations | 2 |
//! | [`fit_quadratic`] | y = a·x² + b·x + c | 3×3 Cramer | 3 |
//! | [`fit_exponential`] | y = a·exp(b·x) | Log-linearized | 2 (y > 0) |
//! | [`fit_logarithmic`] | y = a·ln(x) + b | Linearized | 2 (x > 0) |
//!
//! # Provenance
//!
//! Absorbed from airSpring `metalForge/forge/src/regression.rs` (V009).
//! Validated against Dong et al. (2020) *Agriculture* 10(12):598.

/// Result of a regression fit.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Model name: "linear", "quadratic", "exponential", "logarithmic".
    pub model: &'static str,
    /// Model parameters (interpretation depends on model).
    pub params: Vec<f64>,
    /// Coefficient of determination R².
    pub r_squared: f64,
    /// Root Mean Square Error of the fit.
    pub rmse: f64,
}

impl FitResult {
    /// Slope (first coefficient) — `a` in y = ax + b, y = ax² + bx + c, etc.
    #[must_use]
    pub fn slope(&self) -> Option<f64> {
        self.params.first().copied()
    }

    /// Intercept (constant term) — `b` for linear, `c` for quadratic.
    #[must_use]
    pub fn intercept(&self) -> Option<f64> {
        match self.model {
            "linear" | "exponential" | "logarithmic" => self.params.get(1).copied(),
            "quadratic" => self.params.get(2).copied(),
            _ => None,
        }
    }

    /// All model coefficients as a slice.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.params
    }

    /// Evaluate the fitted model at a single x value.
    #[must_use]
    pub fn predict_one(&self, x: f64) -> Option<f64> {
        match self.model {
            "linear" => Some(self.params[0].mul_add(x, self.params[1])),
            "quadratic" => {
                Some(self.params[0].mul_add(x * x, self.params[1].mul_add(x, self.params[2])))
            }
            "exponential" => Some(self.params[0] * (self.params[1] * x).exp()),
            "logarithmic" => Some(self.params[0].mul_add(x.ln(), self.params[1])),
            _ => None,
        }
    }

    /// Evaluate the fitted model at multiple x values.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> Vec<Option<f64>> {
        x.iter().map(|&xi| self.predict_one(xi)).collect()
    }
}

/// Fit a linear model y = a·x + b via normal equations.
///
/// Returns `None` if fewer than 2 points or the system is singular.
#[must_use]
#[expect(clippy::many_single_char_names, reason = "suppressed")]
pub fn fit_linear(x: &[f64], y: &[f64]) -> Option<FitResult> {
    let n = x.len() as f64;
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }

    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sxy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();

    let det = n.mul_add(sxx, -(sx * sx));
    if det.abs() < 1e-30 {
        return None;
    }

    let a = n.mul_add(sxy, -(sx * sy)) / det;
    let b = sxx.mul_add(sy, -(sx * sxy)) / det;

    let (r2, rmse) = goodness_of_fit(x, y, |xi| a.mul_add(xi, b));
    Some(FitResult {
        model: "linear",
        params: vec![a, b],
        r_squared: r2,
        rmse,
    })
}

/// Fit a quadratic model y = a·x² + b·x + c via 3×3 Cramer's rule.
///
/// Returns `None` if fewer than 3 points or the system is singular.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "math notation"
)]
pub fn fit_quadratic(x: &[f64], y: &[f64]) -> Option<FitResult> {
    let n = x.len() as f64;
    if x.len() < 3 || x.len() != y.len() {
        return None;
    }

    let sx: f64 = x.iter().sum();
    let sx2: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sx3: f64 = x.iter().map(|&xi| xi.powi(3)).sum();
    let sx4: f64 = x.iter().map(|&xi| xi.powi(4)).sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();
    let sx2y: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * xi * yi).sum();

    let m = [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, n]];
    let rhs = [sx2y, sxy, sy];

    let (a, b, c) = cramer_3x3(m, rhs)?;

    let (r2, rmse) = goodness_of_fit(x, y, |xi| a.mul_add(xi * xi, b.mul_add(xi, c)));
    Some(FitResult {
        model: "quadratic",
        params: vec![a, b, c],
        r_squared: r2,
        rmse,
    })
}

/// Fit an exponential model y = a·exp(b·x) via log-linearized least squares.
///
/// Filters to positive y-values. Returns `None` if fewer than 2 valid points.
#[must_use]
pub fn fit_exponential(x: &[f64], y: &[f64]) -> Option<FitResult> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }

    let valid: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|&(_, yi)| *yi > 0.0)
        .map(|(xi, yi)| (*xi, *yi))
        .collect();
    if valid.len() < 2 {
        return None;
    }

    let xv: Vec<f64> = valid.iter().map(|&(xi, _)| xi).collect();
    let ly: Vec<f64> = valid.iter().map(|&(_, yi)| yi.ln()).collect();

    let lin = fit_linear(&xv, &ly)?;
    let b = lin.params[0];
    let a = lin.params[1].exp();

    let yv: Vec<f64> = valid.iter().map(|&(_, yi)| yi).collect();
    let (r2, rmse) = goodness_of_fit(&xv, &yv, |xi| a * (b * xi).exp());
    Some(FitResult {
        model: "exponential",
        params: vec![a, b],
        r_squared: r2,
        rmse,
    })
}

/// Fit a logarithmic model y = a·ln(x) + b via linearized least squares.
///
/// Filters to x > 0.001. Returns `None` if fewer than 2 valid points.
#[must_use]
pub fn fit_logarithmic(x: &[f64], y: &[f64]) -> Option<FitResult> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }

    let valid: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|&(xi, _)| *xi > 0.001)
        .map(|(xi, yi)| (*xi, *yi))
        .collect();
    if valid.len() < 2 {
        return None;
    }

    let lnx: Vec<f64> = valid.iter().map(|&(xi, _)| xi.ln()).collect();
    let yv: Vec<f64> = valid.iter().map(|&(_, yi)| yi).collect();

    let lin = fit_linear(&lnx, &yv)?;
    let a = lin.params[0];
    let b = lin.params[1];

    let xv: Vec<f64> = valid.iter().map(|&(xi, _)| xi).collect();
    let (r2, rmse) = goodness_of_fit(&xv, &yv, |xi| a.mul_add(xi.ln(), b));
    Some(FitResult {
        model: "logarithmic",
        params: vec![a, b],
        r_squared: r2,
        rmse,
    })
}

/// Fit all four models and return those that converge.
#[must_use]
pub fn fit_all(x: &[f64], y: &[f64]) -> Vec<FitResult> {
    [
        fit_linear(x, y),
        fit_quadratic(x, y),
        fit_exponential(x, y),
        fit_logarithmic(x, y),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn goodness_of_fit<F: Fn(f64) -> f64>(x: &[f64], y: &[f64], predict: F) -> (f64, f64) {
    let n = y.len() as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let ss_res: f64 = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| (yi - predict(xi)).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    (r2, (ss_res / n).sqrt())
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    let minor0 = m[1][1].mul_add(m[2][2], -(m[1][2] * m[2][1]));
    let minor1 = m[1][0].mul_add(m[2][2], -(m[1][2] * m[2][0]));
    let minor2 = m[1][0].mul_add(m[2][1], -(m[1][1] * m[2][0]));
    m[0][2].mul_add(minor2, m[0][0].mul_add(minor0, -(m[0][1] * minor1)))
}

fn cramer_3x3(m: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<(f64, f64, f64)> {
    let d = det3(&m);
    if d.abs() < 1e-30 {
        return None;
    }
    let inv = 1.0 / d;

    let m0 = [
        [rhs[0], m[0][1], m[0][2]],
        [rhs[1], m[1][1], m[1][2]],
        [rhs[2], m[2][1], m[2][2]],
    ];
    let m1 = [
        [m[0][0], rhs[0], m[0][2]],
        [m[1][0], rhs[1], m[1][2]],
        [m[2][0], rhs[2], m[2][2]],
    ];
    let m2 = [
        [m[0][0], m[0][1], rhs[0]],
        [m[1][0], m[1][1], rhs[1]],
        [m[2][0], m[2][1], rhs[2]],
    ];

    Some((inv * det3(&m0), inv * det3(&m1), inv * det3(&m2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_perfect() {
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0f64.mul_add(xi, 1.0)).collect();
        let r = fit_linear(&x, &y).unwrap();
        assert_eq!(r.model, "linear");
        assert!((r.params[0] - 2.0).abs() < 1e-10);
        assert!((r.params[1] - 1.0).abs() < 1e-10);
        assert!((r.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_perfect() {
        let x: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (0.5 * xi).mul_add(xi, (-2.0f64).mul_add(xi, 3.0)))
            .collect();
        let r = fit_quadratic(&x, &y).unwrap();
        assert_eq!(r.model, "quadratic");
        assert!((r.params[0] - 0.5).abs() < 1e-6);
        assert!((r.params[1] + 2.0).abs() < 1e-6);
        assert!((r.params[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_fit() {
        let x: Vec<f64> = (1..=10).map(|i| f64::from(i) * 0.3).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (xi * 0.5).exp()).collect();
        let r = fit_exponential(&x, &y).unwrap();
        assert_eq!(r.model, "exponential");
        assert!((r.params[0] - 2.0).abs() < 0.1);
        assert!((r.params[1] - 0.5).abs() < 0.05);
        assert!(r.r_squared > 0.99);
    }

    #[test]
    fn test_logarithmic_fit() {
        let x: Vec<f64> = (1..=15).map(|i| f64::from(i).mul_add(0.5, 0.1)).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0f64.mul_add(xi.ln(), 1.0)).collect();
        let r = fit_logarithmic(&x, &y).unwrap();
        assert_eq!(r.model, "logarithmic");
        assert!((r.params[0] - 3.0).abs() < 0.01);
        assert!((r.params[1] - 1.0).abs() < 0.01);
        assert!(r.r_squared > 0.999);
    }

    #[test]
    fn test_fit_all_returns_models() {
        let x: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.5f64.mul_add(xi, 0.3)).collect();
        let models = fit_all(&x, &y);
        assert!(models.len() >= 2, "Got {} models", models.len());
        let linear = models.iter().find(|r| r.model == "linear").unwrap();
        assert!(linear.r_squared > 0.99);
    }

    #[test]
    fn test_predict_linear() {
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0f64.mul_add(xi, 2.0)).collect();
        let r = fit_linear(&x, &y).unwrap();
        let pred = r.predict(&[0.0, 5.0, 10.0]);
        assert!((pred[0].unwrap() - 2.0).abs() < 1e-8);
        assert!((pred[1].unwrap() - 17.0).abs() < 1e-8);
        assert!((pred[2].unwrap() - 32.0).abs() < 1e-8);
    }

    #[test]
    fn test_predict_one_quadratic() {
        let x: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let r = fit_quadratic(&x, &y).unwrap();
        assert!((r.predict_one(3.0).unwrap() - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_predict_one_unknown_model() {
        let r = FitResult {
            model: "unknown",
            params: vec![1.0, 2.0],
            r_squared: 0.0,
            rmse: 0.0,
        };
        assert!(r.predict_one(1.0).is_none());
    }

    #[test]
    fn test_insufficient_points() {
        assert!(fit_linear(&[1.0], &[2.0]).is_none());
        assert!(fit_quadratic(&[1.0, 2.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_singular_system() {
        assert!(fit_linear(&[5.0, 5.0, 5.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn test_exponential_negative_y() {
        assert!(fit_exponential(&[1.0, 2.0, 3.0], &[-1.0, -2.0, -3.0]).is_none());
    }

    #[test]
    fn test_logarithmic_negative_x() {
        assert!(fit_logarithmic(&[-1.0, -2.0, -3.0], &[1.0, 2.0, 3.0]).is_none());
    }
}
