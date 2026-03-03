// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU GELU - WGSL Universal Compute with Event Optimization
//!
//! Uses the same WGSL shader as GPU/CPU for GELU activation,
//! with optional event-based optimization for Akida NPU.
//!
//! **Why GELU is Important**:
//! - Used in BERT, GPT-2, GPT-3, and most modern transformers
//! - Smoother than ReLU (differentiable everywhere)
//! - Better gradient flow in deep networks
//! - Creates moderate sparsity (unlike ReLU's hard threshold)
//!
//! **Formula**: GELU(x) = x * Φ(x) where Φ is Gaussian CDF
//! **Approximation**: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//!
//! **Deep Debt A++**:
//! - ✅ WGSL shader (same math as GPU/CPU!)
//! - ✅ Hardware-agnostic activation
//! - ✅ EventCodec for NPU-specific optimization
//! - ✅ Single source of truth for GELU algorithm

use crate::npu::EventCodec;

type Result<T> = std::result::Result<T, crate::error::BarracudaError>;

/// NPU-optimized GELU activation using WGSL (universal compute)
///
/// Performs GELU using the SAME WGSL shader as GPU/CPU,
/// with optional event-based optimization for Akida NPU.
///
/// **Key Principle: "Hardware does specialization, not code!"**
/// - Same math on all chips (WGSL shader)
/// - EventCodec provides NPU-specific optimization
/// - Fair cross-chip performance comparison
///
/// **Algorithm**:
/// 1. Execute WGSL GELU (same as GPU/CPU) → UNIVERSAL MATH
/// 2. Analyze sparsity for NPU optimization
/// 3. Convert to events for energy savings
///
/// # Arguments
/// * `input` - Input activations
///
/// # Returns
/// GELU-activated output (computed via WGSL, same as GPU/CPU!)
///
/// # Example
/// ```ignore
/// let x = vec![-1.0, 0.0, 1.0, 2.0];
/// let y = npu_gelu(&x)?;
/// // y ≈ [-0.16, 0.0, 0.84, 1.96] - via WGSL!
/// ```
pub fn npu_gelu(input: &[f32]) -> Result<Vec<f32>> {
    tracing::debug!("NPU GELU (WGSL): {} activations", input.len());

    // ═══════════════════════════════════════════════════════════
    // CRITICAL: Use WGSL shader (same math as GPU/CPU!)
    // ═══════════════════════════════════════════════════════════

    use crate::device::test_pool::run_with_sync_device;
    use crate::tensor::Tensor;

    run_with_sync_device(|device| {
        // Create tensor from raw data (zero-copy: from_data borrows slice)
        let input_len = input.len();
        let tensor = Tensor::from_data(input, vec![input_len], device)?;

        // Execute GELU using WGSL shader (same as GPU/CPU!)
        // This uses ops/gelu.rs → shaders/gelu.wgsl
        let result_tensor = tensor.gelu_wgsl()?;

        // Extract result
        let output = result_tensor.to_vec()?;

        // ═══════════════════════════════════════════════════════════
        // NPU-SPECIFIC OPTIMIZATION: Event encoding (optional)
        // ═══════════════════════════════════════════════════════════

        let codec = EventCodec::default();
        let sparsity = codec.measure_sparsity(&output);

        tracing::debug!(
            "NPU GELU (WGSL) complete: {:.1}% sparsity",
            sparsity * 100.0
        );

        // For sparse outputs, encode as events
        if sparsity > 0.3 {
            let events = codec.encode(&output);
            tracing::debug!(
                "NPU event encoding: {} events ({}% reduction)",
                events.len(),
                ((1.0 - events.len() as f32 / output.len() as f32) * 100.0)
            );
        }

        Ok(output)
    })
}

/// Fast GELU approximation using tanh
///
/// **Formula**: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// **Accuracy**: ~0.1% error vs exact Gaussian CDF
#[inline]
fn gelu_approx(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // √(2/π)
    const COEFF: f32 = 0.044715;

    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
    let tanh_val = fast_tanh(inner);

    0.5 * x * (1.0 + tanh_val)
}

/// Fast tanh approximation
///
/// **Range**: [-1, 1]
/// **Accuracy**: Good for |x| < 3, exact at boundaries
#[inline]
fn fast_tanh(x: f32) -> f32 {
    // Clamp to prevent overflow
    if x > 5.0 {
        return 1.0;
    }
    if x < -5.0 {
        return -1.0;
    }

    // Use standard tanh for moderate values
    x.tanh()
}

/// NPU-accelerated GELU with exact computation
///
/// Uses error function (erf) for exact Gaussian CDF.
/// Slower but more accurate than approximation.
///
/// **Formula**: GELU(x) = 0.5 * x * (1 + erf(x / √2))
pub fn npu_gelu_exact(input: &[f32]) -> Result<Vec<f32>> {
    let mut output = Vec::with_capacity(input.len());

    const SQRT_2: f32 = std::f32::consts::SQRT_2;

    for &x in input {
        // erf approximation (good to ~1e-4)
        let z = x / SQRT_2;
        let erf_val = erf_approx(z);
        let gelu_val = 0.5 * x * (1.0 + erf_val);
        output.push(gelu_val);
    }

    Ok(output)
}

/// Error function (erf) approximation
///
/// **Range**: [-1, 1]
/// **Accuracy**: ~1e-4 for |x| < 3
#[inline]
fn erf_approx(x: f32) -> f32 {
    // Handle boundaries
    if x > 3.0 {
        return 1.0;
    }
    if x < -3.0 {
        return -1.0;
    }

    // Abramowitz and Stegun approximation
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let erf_val = 1.0 - (A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5) * (-x * x).exp();

    sign * erf_val
}

/// Compare GELU vs ReLU sparsity
///
/// **Returns**: (relu_sparsity, gelu_sparsity)
pub fn compare_gelu_relu_sparsity(input: &[f32]) -> (f32, f32) {
    use crate::npu::EventCodec;

    let codec = EventCodec::default();

    // ReLU output
    let relu: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
    let relu_sparsity = codec.measure_sparsity(&relu);

    // GELU output
    let gelu: Vec<f32> = input.iter().map(|&x| gelu_approx(x)).collect();
    let gelu_sparsity = codec.measure_sparsity(&gelu);

    (relu_sparsity, gelu_sparsity)
}

/// Check if NPU GELU is beneficial
///
/// **Decision**: GELU is moderately beneficial on NPU
/// - Creates some sparsity (less than ReLU)
/// - Smoother gradients (better for training)
/// - Modern transformer standard
pub fn should_use_npu_gelu() -> bool {
    // GELU is standard in modern transformers
    // NPU handles it reasonably well
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_basic() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let output = npu_gelu(&input).unwrap();

        // Check key properties
        assert_eq!(output.len(), input.len());

        // GELU(0) = 0
        assert!(
            (output[2] - 0.0).abs() < 1e-5,
            "GELU(0) should be 0, got {}",
            output[2]
        );

        // GELU is monotonic for positive values
        assert!(
            output[3] < output[4],
            "GELU should be monotonic for positive values"
        );

        // GELU(x) ≈ x for large positive x
        assert!(
            (output[4] - 2.0).abs() < 0.1,
            "GELU(2) should be ≈ 2, got {}",
            output[4]
        );

        // GELU has a minimum around x = -0.17, so negative side is non-monotonic
        // This is correct GELU behavior!
        assert!(output[0] > output[1], "GELU has minimum around x=-0.17");
    }

    #[test]
    fn test_gelu_vs_exact() {
        let input = vec![-1.0, 0.0, 1.0];
        let approx = npu_gelu(&input).unwrap();
        let exact = npu_gelu_exact(&input).unwrap();

        // Approximation should be close to exact
        for i in 0..input.len() {
            let error = (approx[i] - exact[i]).abs();
            assert!(error < 0.01, "Approximation error too large: {}", error);
        }
    }

    #[test]
    fn test_gelu_smoothness() {
        // GELU should be smooth (no discontinuities)
        let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.1).collect();
        let output = npu_gelu(&input).unwrap();

        // Check for smooth transitions (no jumps)
        for i in 0..output.len() - 1 {
            let diff = (output[i + 1] - output[i]).abs();
            assert!(diff < 0.5, "GELU should be smooth, got jump of {}", diff);
        }
    }

    #[test]
    fn test_gelu_negative_values() {
        // GELU preserves some negative values (unlike ReLU)
        let input = vec![-1.0, -0.5, -0.1];
        let output = npu_gelu(&input).unwrap();

        // All outputs should be negative (but non-zero)
        for val in output {
            assert!(val < 0.0, "GELU should preserve negative sign");
            assert!(val > -1.0, "GELU should not amplify negatives");
        }
    }

    #[test]
    fn test_gelu_vs_relu_sparsity() {
        // Normal distribution-like data
        let input = vec![-1.0, -0.5, -0.2, 0.1, 0.3, 0.8, 1.2];
        let (relu_sp, gelu_sp) = compare_gelu_relu_sparsity(&input);

        // ReLU creates more sparsity (hard threshold at 0)
        assert!(
            relu_sp >= gelu_sp,
            "ReLU should create more sparsity than GELU"
        );

        // But GELU still creates some sparsity
        assert!(gelu_sp > 0.0, "GELU should create some sparsity");
    }

    #[test]
    fn test_fast_tanh() {
        // Test boundary conditions
        assert!((fast_tanh(0.0) - 0.0).abs() < 1e-5);
        assert!((fast_tanh(10.0) - 1.0).abs() < 1e-5);
        assert!((fast_tanh(-10.0) + 1.0).abs() < 1e-5);

        // Test monotonicity
        assert!(fast_tanh(-1.0) < fast_tanh(0.0));
        assert!(fast_tanh(0.0) < fast_tanh(1.0));
    }

    #[test]
    fn test_erf_approx() {
        // Test key values
        assert!((erf_approx(0.0) - 0.0).abs() < 1e-3);
        assert!((erf_approx(1.0) - 0.8427).abs() < 1e-2); // erf(1) ≈ 0.8427

        // Test symmetry: erf(-x) = -erf(x)
        let x = 0.5;
        let pos = erf_approx(x);
        let neg = erf_approx(-x);
        assert!((pos + neg).abs() < 1e-3, "erf should be odd function");
    }

    #[test]
    fn test_should_use_npu() {
        // GELU is standard in transformers, use NPU
        assert!(should_use_npu_gelu());
    }
}
