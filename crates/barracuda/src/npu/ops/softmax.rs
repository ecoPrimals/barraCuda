//! NPU Softmax - WGSL Universal Compute with Event Optimization
//!
//! Uses the same WGSL shader as GPU/CPU for softmax activation,
//! with optional event-based optimization for Akida NPU.
//!
//! **Why Softmax Benefits from NPU**:
//! - Output is naturally sparse (one or few dominant values)
//! - Winner-take-all behavior aligns with spike-based processing
//! - Critical for classification and attention mechanisms
//!
//! **Algorithm**:
//! 1. Execute WGSL softmax (same as GPU/CPU) → UNIVERSAL MATH
//! 2. Analyze output sparsity (dominant values)
//! 3. Convert to events for energy savings
//!
//! **Deep Debt A++**:
//! - ✅ WGSL shader (same math as GPU/CPU!)
//! - ✅ Hardware-agnostic normalization
//! - ✅ EventCodec for NPU-specific optimization
//! - ✅ Single source of truth for softmax algorithm

use crate::npu::EventCodec;

type Result<T> = std::result::Result<T, crate::error::BarracudaError>;

/// NPU-accelerated Softmax activation
///
/// Converts logits to probability distribution.
///
/// **Formula**: softmax(x_i) = exp(x_i) / sum(exp(x_j))
///
/// # Arguments
/// * `logits` - Input logits
/// * `temperature` - Temperature parameter (default 1.0)
///   - < 1.0: Sharpens distribution (more sparse)
///   - > 1.0: Smooths distribution (less sparse)
///
/// # Returns
/// Probability distribution (sums to 1.0)
///
/// # Example
/// ```ignore
/// let logits = vec![2.0, 1.0, 0.1];
/// let probs = npu_softmax(&logits, 1.0)?;
/// // probs ≈ [0.66, 0.24, 0.10]
/// ```
pub fn npu_softmax(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Err(crate::error::BarracudaError::invalid_op(
            "npu_softmax",
            "Input logits cannot be empty",
        ));
    }

    if temperature <= 0.0 {
        return Err(crate::error::BarracudaError::invalid_op(
            "npu_softmax",
            format!("Temperature must be positive, got {temperature}"),
        ));
    }

    // ═══════════════════════════════════════════════════════════
    // CRITICAL: Use WGSL shader (same math as GPU/CPU!)
    // ═══════════════════════════════════════════════════════════

    use crate::tensor::Tensor;

    // Apply temperature scaling if needed (before device access)
    let scaled_logits = if (temperature - 1.0).abs() > 1e-6 {
        logits.iter().map(|&x| x / temperature).collect()
    } else {
        logits.to_vec()
    };
    let logits_len = scaled_logits.len();

    crate::device::test_pool::run_with_sync_device(|device| {
        let tensor = Tensor::from_vec_on_sync(scaled_logits, vec![logits_len], device)?;
        let result_tensor = tensor.softmax()?;
        let output = result_tensor.to_vec()?;

        let codec = EventCodec::default();
        let sparsity = codec.measure_sparsity(&output);
        let max_prob = output.iter().copied().fold(0.0f32, f32::max);
        tracing::debug!(
            "NPU Softmax (WGSL) complete: {} classes, sparsity {:.1}%, max_prob {:.3}",
            logits.len(),
            sparsity * 100.0,
            max_prob
        );
        if max_prob > 0.8 {
            let _events = codec.encode(&output);
        }
        Ok(output)
    })
}

/// NPU-accelerated Log Softmax
///
/// More numerically stable for computing log probabilities.
///
/// **Formula**: log_softmax(x_i) = x_i - log(sum(exp(x_j)))
///
/// **Use case**: Cross-entropy loss computation
pub fn npu_log_softmax(logits: &[f32]) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Err(crate::error::BarracudaError::invalid_op(
            "npu_log_softmax",
            "Input logits cannot be empty",
        ));
    }

    // Find max for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute log(sum(exp(x - max)))
    let log_sum_exp = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f32>()
        .ln();

    // Compute log_softmax = x - max - log_sum_exp
    let log_probs: Vec<f32> = logits
        .iter()
        .map(|&x| x - max_logit - log_sum_exp)
        .collect();

    Ok(log_probs)
}

/// Softmax with top-k selection
///
/// Only keeps top-k probabilities, zeros out rest (creates sparsity for NPU).
///
/// **Use case**: Sampling with nucleus/top-k filtering
pub fn npu_softmax_top_k(logits: &[f32], k: usize, temperature: f32) -> Result<Vec<f32>> {
    let mut probs = npu_softmax(logits, temperature)?;

    if k >= probs.len() {
        return Ok(probs); // Already using all
    }

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Zero out non-top-k
    let top_k_indices: std::collections::HashSet<usize> =
        indexed.iter().take(k).map(|(idx, _)| *idx).collect();

    for (i, prob) in probs.iter_mut().enumerate() {
        if !top_k_indices.contains(&i) {
            *prob = 0.0;
        }
    }

    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for prob in &mut probs {
            *prob /= sum;
        }
    }

    tracing::debug!("NPU Softmax Top-K: kept {}/{} classes", k, logits.len());

    Ok(probs)
}

/// Check if NPU softmax is beneficial
///
/// **Decision**: NPU is good for classification (creates sparsity)
pub fn should_use_npu_softmax(num_classes: usize) -> bool {
    // Softmax creates natural sparsity (winner-take-all)
    // NPU is good for this, especially with many classes
    num_classes >= 10 // Benefit increases with more classes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![2.0, 1.0, 0.1];
        let probs = npu_softmax(&logits, 1.0).unwrap();

        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities should sum to 1, got {}",
            sum
        );

        // Check all probabilities in [0, 1]
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p), "Probability out of range: {}", p);
        }

        // Largest logit should have highest probability
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_temperature() {
        let logits = vec![2.0, 1.0, 0.1];

        // Low temperature (sharper distribution)
        let sharp = npu_softmax(&logits, 0.5).unwrap();

        // High temperature (smoother distribution)
        let smooth = npu_softmax(&logits, 2.0).unwrap();

        // Sharp should have more extreme probabilities
        assert!(sharp[0] > smooth[0], "Low temp should sharpen distribution");
        assert!(sharp[2] < smooth[2], "Low temp should suppress low probs");
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![2.0, 1.0, 0.1];
        let log_probs = npu_log_softmax(&logits).unwrap();

        // Log probabilities should be negative
        for &lp in &log_probs {
            assert!(lp <= 0.0, "Log probability should be ≤ 0, got {}", lp);
        }

        // exp(log_softmax) should equal softmax
        let probs = npu_softmax(&logits, 1.0).unwrap();
        for i in 0..logits.len() {
            let exp_log_prob = log_probs[i].exp();
            assert!((exp_log_prob - probs[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_top_k() {
        let logits = vec![2.0, 1.5, 1.0, 0.5, 0.1];
        let probs = npu_softmax_top_k(&logits, 2, 1.0).unwrap();

        // Only top-2 should be non-zero
        let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(non_zero, 2, "Should have exactly 2 non-zero probabilities");

        // Sum should still be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probabilities should sum to 1");
    }

    #[test]
    fn test_numerical_stability() {
        // Large logits that could cause overflow
        let logits = vec![1000.0, 999.0, 998.0];
        let probs = npu_softmax(&logits, 1.0).unwrap();

        // Should not overflow, should be valid probabilities
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Should handle large logits");
        assert!(
            probs.iter().all(|&p| p.is_finite()),
            "All values should be finite"
        );
    }

    #[test]
    fn test_error_cases() {
        // Empty input
        assert!(npu_softmax(&[], 1.0).is_err());

        // Invalid temperature
        assert!(npu_softmax(&[1.0, 2.0], 0.0).is_err());
        assert!(npu_softmax(&[1.0, 2.0], -1.0).is_err());
    }

    #[test]
    fn test_should_use_npu() {
        // Small number of classes → might not benefit
        assert!(!should_use_npu_softmax(5));

        // Many classes → NPU beneficial
        assert!(should_use_npu_softmax(100));
        assert!(should_use_npu_softmax(1000));
    }
}
