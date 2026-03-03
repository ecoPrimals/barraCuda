//! NPU weight export for ESN readout layer.
//!
//! Provides affine int8 quantization for deploying ESN readout weights to
//! NPU hardware (e.g. Akida AKD1000) that expects quantized weights.

/// Int8-quantized readout weights for NPU deployment.
///
/// The readout layer computes `output = W_out @ state` where W_out has shape
/// `[output_dim, input_dim]` (output_dim × reservoir_size). Quantized for
/// efficient int8 matrix-vector multiply on NPU.
#[derive(Debug, Clone)]
pub struct NpuReadoutWeights {
    /// Quantized readout weights: `output_dim × input_dim`, row-major.
    pub weights_i8: Vec<i8>,
    /// Quantization scale: `real ≈ (quantized - zero_point) * scale`.
    pub scale: f64,
    /// Quantization zero point (integer).
    pub zero_point: i64,
    /// Reservoir dimensionality (input to readout = state size).
    pub input_dim: usize,
    /// Number of output features.
    pub output_dim: usize,
}

/// Affine quantization of f64 values to int8.
///
/// Uses the formula: `q = round(x / scale) + zero_point`
/// with `scale = (max - min) / 254.0` and `zero_point = round(-min / scale) - 127`.
/// Dequantization: `x ≈ (q - zero_point) * scale`.
///
/// # Returns
/// `(quantized_values, scale, zero_point)`
///
/// # Example
/// ```
/// use barracuda::esn_v2::quantize_affine_i8_f64;
///
/// let values = vec![0.0, 1.0, 2.0, -1.0];
/// let (quantized, scale, zero_point) = quantize_affine_i8_f64(&values);
/// // Dequantize: (q - zero_point) * scale ≈ original
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn quantize_affine_i8_f64(values: &[f64]) -> (Vec<i8>, f64, i64) {
    if values.is_empty() {
        return (Vec::new(), 1.0, 0);
    }

    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let range = max_val - min_val;
    let scale = if range > 0.0 { range / 254.0 } else { 1.0 };
    let zero_point = (round(-min_val / scale) - 127.0) as i64;

    let quantized: Vec<i8> = values
        .iter()
        .map(|&x| {
            let q_f64 = round(x / scale) + zero_point as f64;
            let q = round(q_f64).clamp(-128.0, 127.0) as i64;
            q.clamp(-128, 127) as i8
        })
        .collect();

    (quantized, scale, zero_point)
}

#[inline]
fn round(x: f64) -> f64 {
    (x + 0.5_f64.copysign(x)).trunc()
}

/// Dequantize int8 values back to f64.
///
/// `x = (q - zero_point) * scale`
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn dequantize_affine_i8_f64(quantized: &[i8], scale: f64, zero_point: i64) -> Vec<f64> {
    quantized
        .iter()
        .map(|&q| (q as i64 - zero_point) as f64 * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_round_trip_max_error() {
        // Round-trip: quantize then dequantize, verify max error < 1% of range
        let values: Vec<f64> = (0..100).map(|i| -5.0 + (i as f64 * 0.15)).collect();
        let (quantized, scale, zero_point) = quantize_affine_i8_f64(&values);
        let dequantized = dequantize_affine_i8_f64(&quantized, scale, zero_point);

        let range = values
            .iter()
            .cloned()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| {
                (a.min(x), b.max(x))
            });
        let range_size = range.1 - range.0;
        let max_error = values
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_error < 0.01 * range_size,
            "max_error {} should be < 1% of range {}",
            max_error,
            0.01 * range_size
        );
    }

    #[test]
    fn test_quantize_known_values() {
        let values = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        let (quantized, scale, zero_point) = quantize_affine_i8_f64(&values);
        let dequantized = dequantize_affine_i8_f64(&quantized, scale, zero_point);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            let err = (orig - deq).abs();
            assert!(err < 0.1, "orig {} deq {} err {}", orig, deq, err);
        }
    }

    #[test]
    fn test_quantize_all_zero() {
        let values = vec![0.0; 10];
        let (quantized, scale, zero_point) = quantize_affine_i8_f64(&values);
        assert_eq!(quantized.len(), 10);
        assert!(quantized.iter().all(|&q| q == quantized[0]));
        let dequantized = dequantize_affine_i8_f64(&quantized, scale, zero_point);
        assert!(dequantized.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_quantize_single_value() {
        // When all values are identical, range=0 → scale=1, so we quantize to nearest int.
        // Dequantized will be integer; expect error up to 0.5.
        let values = vec![1.5; 5];
        let (quantized, scale, zero_point) = quantize_affine_i8_f64(&values);
        assert_eq!(quantized.len(), 5);
        let dequantized = dequantize_affine_i8_f64(&quantized, scale, zero_point);
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 1.0,
                "single-value quantize: orig {} deq {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_quantize_large_range() {
        let values: Vec<f64> = (-50..50).map(|i| i as f64 * 10.0).collect();
        let (quantized, scale, zero_point) = quantize_affine_i8_f64(&values);
        let dequantized = dequantize_affine_i8_f64(&quantized, scale, zero_point);

        let range = 1000.0; // -500 to 500
        let max_error = values
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(max_error < 0.01 * range);
    }

    #[test]
    fn test_quantize_empty() {
        let values: Vec<f64> = vec![];
        let (quantized, scale, zero_point) = quantize_affine_i8_f64(&values);
        assert!(quantized.is_empty());
        assert!((scale - 1.0).abs() < 1e-10);
        assert_eq!(zero_point, 0);
    }
}
