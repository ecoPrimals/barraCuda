// SPDX-License-Identifier: AGPL-3.0-or-later
//! Activation functions for neural networks and signal processing.
//!
//! **Shader-first architecture**: batch functions (`relu_batch`, `sigmoid_batch`,
//! `gelu_batch`, `swish_batch`) dispatch through WGSL via naga-exec when the
//! `cpu-shader` feature is enabled, using the same shader math that runs on GPU.
//! Scalar functions remain native Rust.
//!
//! For GPU activation ops, see `ops::relu`, `ops::sigmoid`, `ops::gelu_wgsl`, etc.

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
        x.exp().ln_1p()
    }
}

/// Leaky `ReLU`: `max(alpha * x, x)` with default `alpha = 0.01`
#[inline]
#[must_use]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 { x } else { alpha * x }
}

// ── Batch functions ──────────────────────────────────────────────────────────
//
// When the `cpu-shader` feature is enabled, batch functions dispatch through
// WGSL via naga-exec — the same shader that runs on GPU. Falls back to
// scalar Rust if the shader fails or the feature is disabled.

/// Apply `ReLU` element-wise to a slice.
///
/// With `cpu-shader`, dispatches through `relu_f64.wgsl` via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn relu_batch(input: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        let wgsl = include_str!("shaders/activation/relu_f64.wgsl");
        if let Ok(out) = crate::unified_hardware::shader_batch_unary_f64(wgsl, "main", input) {
            return out;
        }
    }
    input.iter().map(|&x| relu(x)).collect()
}

#[cfg(feature = "cpu-shader")]
const SIGMOID_F64_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    let x = input[idx];
    if x >= 0.0 {
        output[idx] = 1.0 / (1.0 + exp(-x));
    } else {
        let ez = exp(x);
        output[idx] = ez / (1.0 + ez);
    }
}
";

/// Apply sigmoid element-wise to a slice.
///
/// With `cpu-shader`, dispatches through inline WGSL via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn sigmoid_batch(input: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        if let Ok(out) =
            crate::unified_hardware::shader_batch_unary_f64(SIGMOID_F64_WGSL, "main", input)
        {
            return out;
        }
    }
    input.iter().map(|&x| sigmoid(x)).collect()
}

#[cfg(feature = "cpu-shader")]
const GELU_F64_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    let x = input[idx];
    let k = 0.7978845608028654; // sqrt(2/pi)
    let inner = k * fma(0.044715 * x * x, x, x);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
";

/// Apply GELU element-wise to a slice.
///
/// With `cpu-shader`, dispatches through inline WGSL via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn gelu_batch(input: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        if let Ok(out) =
            crate::unified_hardware::shader_batch_unary_f64(GELU_F64_WGSL, "main", input)
        {
            return out;
        }
    }
    input.iter().map(|&x| gelu(x)).collect()
}

#[cfg(feature = "cpu-shader")]
const SWISH_F64_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    let x = input[idx];
    let s = 1.0 / (1.0 + exp(-x));
    output[idx] = x * s;
}
";

/// Apply swish element-wise to a slice.
///
/// With `cpu-shader`, dispatches through inline WGSL via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn swish_batch(input: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        if let Ok(out) =
            crate::unified_hardware::shader_batch_unary_f64(SWISH_F64_WGSL, "main", input)
        {
            return out;
        }
    }
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
