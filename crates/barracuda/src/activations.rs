// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical CPU activation functions for neural networks and signal processing.
//!
//! These are the single-source-of-truth scalar and batch implementations.
//! Springs should import `barracuda::activations::*` instead of reimplementing.
//!
//! For GPU activation ops, see `ops::relu`, `ops::sigmoid`, `ops::gelu_wgsl`, etc.
//!
//! Created in response to neuralSpring S134 request — 7 duplicate activation
//! functions across springs consolidated here.

use std::f64::consts::PI;

// ── Scalar functions ─────────────────────────────────────────────────────────

/// `ReLU`: `max(0, x)`
#[inline]
#[must_use]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Sigmoid: `1 / (1 + exp(-x))`
///
/// Numerically stable: uses `exp(x) / (1 + exp(x))` for negative inputs
/// to avoid overflow in `exp(-x)`.
#[inline]
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ez = x.exp();
        ez / (1.0 + ez)
    }
}

/// `GELU` (Gaussian Error Linear Unit): `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
///
/// Uses the tanh approximation from the original paper (Hendrycks & Gimpel, 2016).
#[inline]
#[must_use]
pub fn gelu(x: f64) -> f64 {
    let inner = (2.0 / PI).sqrt() * 0.044_715_f64.mul_add(x.powi(3), x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Swish / `SiLU`: `x * sigmoid(x)`
#[inline]
#[must_use]
pub fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Mish: `x * tanh(softplus(x))` where `softplus(x) = ln(1 + exp(x))`
#[inline]
#[must_use]
pub fn mish(x: f64) -> f64 {
    x * softplus(x).tanh()
}

/// Softplus: `ln(1 + exp(x))`
///
/// Numerically stable: uses `x` directly for large positive inputs.
#[inline]
#[must_use]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Leaky `ReLU`: `max(alpha * x, x)` with default `alpha = 0.01`
#[inline]
#[must_use]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 { x } else { alpha * x }
}

// ── Batch functions ──────────────────────────────────────────────────────────

/// Apply `ReLU` element-wise to a slice.
#[must_use]
pub fn relu_batch(input: &[f64]) -> Vec<f64> {
    input.iter().map(|&x| relu(x)).collect()
}

/// Apply sigmoid element-wise to a slice.
#[must_use]
pub fn sigmoid_batch(input: &[f64]) -> Vec<f64> {
    input.iter().map(|&x| sigmoid(x)).collect()
}

/// Apply GELU element-wise to a slice.
#[must_use]
pub fn gelu_batch(input: &[f64]) -> Vec<f64> {
    input.iter().map(|&x| gelu(x)).collect()
}

/// Apply swish element-wise to a slice.
#[must_use]
pub fn swish_batch(input: &[f64]) -> Vec<f64> {
    input.iter().map(|&x| swish(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_basic() {
        assert!((relu(3.0) - 3.0).abs() < f64::EPSILON);
        assert!((relu(-1.0)).abs() < f64::EPSILON);
        assert!((relu(0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-14);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn sigmoid_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 5.0, 10.0] {
            let sum = sigmoid(x) + sigmoid(-x);
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "sigmoid(x) + sigmoid(-x) = 1 failed at x={x}"
            );
        }
    }

    #[test]
    fn gelu_basic() {
        assert!(gelu(0.0).abs() < 1e-14, "GELU(0) = 0");
        assert!(gelu(10.0) > 9.99, "GELU(large) ≈ x");
        assert!(gelu(-10.0).abs() < 0.01, "GELU(large negative) ≈ 0");
    }

    #[test]
    fn swish_basic() {
        assert!(swish(0.0).abs() < 1e-14, "swish(0) = 0");
        assert!((swish(1.0) - sigmoid(1.0)).abs() < 1e-14);
    }

    #[test]
    fn softplus_basic() {
        assert!((softplus(0.0) - (2.0_f64).ln()).abs() < 1e-14);
        assert!((softplus(100.0) - 100.0).abs() < 1e-10);
        assert!(softplus(-100.0) < 1e-40);
    }

    #[test]
    fn leaky_relu_basic() {
        assert!((leaky_relu(1.0, 0.01) - 1.0).abs() < f64::EPSILON);
        assert!((leaky_relu(-1.0, 0.01) - (-0.01)).abs() < 1e-14);
    }

    #[test]
    fn batch_lengths() {
        let input = vec![1.0, -1.0, 0.0, 2.0];
        assert_eq!(relu_batch(&input).len(), 4);
        assert_eq!(sigmoid_batch(&input).len(), 4);
        assert_eq!(gelu_batch(&input).len(), 4);
        assert_eq!(swish_batch(&input).len(), 4);
    }
}
