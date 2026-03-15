// SPDX-License-Identifier: AGPL-3.0-only
//! Event Codec — Dense ↔ Sparse Event Conversion.
//!
//! Converts between dense tensor representations and sparse event streams
//! for NPU execution. Sparse encoding is where NPU energy efficiency comes
//! from — only above-threshold activations are transmitted.

use bytes::{Bytes, BytesMut};

/// Event codec for NPU dense/sparse conversion.
pub struct EventCodec {
    threshold: f32,
}

impl EventCodec {
    /// Create a codec with the given activation threshold (0.0–1.0).
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Convert dense activations to sparse events (index + value pairs).
    ///
    /// Returns `Bytes` for zero-copy sharing with downstream consumers.
    /// Each event is 5 bytes: 4-byte LE index + 1-byte quantized value.
    #[must_use]
    pub fn encode(&self, input: &[f32]) -> Bytes {
        let mut buf = BytesMut::with_capacity(input.len() * 5);
        for (idx, &val) in input.iter().enumerate() {
            if val > self.threshold {
                buf.extend_from_slice(&(idx as u32).to_le_bytes());
                buf.extend_from_slice(&[(val * 255.0).clamp(0.0, 255.0) as u8]);
            }
        }
        buf.freeze()
    }

    /// Convert dense to events (simplified sequential encoding).
    ///
    /// For small inputs (e.g. MNIST 784 dims) where indices are implicit.
    /// Returns `Bytes` for zero-copy sharing.
    #[must_use]
    pub fn encode_simple(&self, input: &[f32]) -> Bytes {
        let buf: Vec<u8> = input
            .iter()
            .filter(|&&val| val > self.threshold)
            .map(|&val| (val * 255.0).clamp(0.0, 255.0) as u8)
            .collect();
        Bytes::from(buf)
    }

    /// Convert sparse events back to dense
    ///
    /// **Deep Debt**: Safe reconstruction, no unsafe
    ///
    /// # Arguments
    /// * `events` - Sparse event stream
    /// * `size` - Output size
    ///
    /// # Returns
    /// Dense tensor reconstruction
    #[must_use]
    pub fn decode(&self, events: &[u8], size: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; size];

        // Parse events: [idx(4 bytes), val(1 byte)]*
        let mut i = 0;
        while i + 4 < events.len() {
            let idx_bytes = [events[i], events[i + 1], events[i + 2], events[i + 3]];
            let idx = u32::from_le_bytes(idx_bytes) as usize;
            let val = events[i + 4];

            if idx < size {
                dense[idx] = (val as f32) / 255.0;
            }

            i += 5;
        }

        dense
    }

    /// Decode simple event encoding
    ///
    /// For simple encoding (values only, sequential indices)
    #[must_use]
    pub fn decode_simple(&self, events: &[u8], size: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; size];

        for (idx, &event) in events.iter().enumerate() {
            if idx < size {
                dense[idx] = (event as f32) / 255.0;
            }
        }

        dense
    }

    /// Calculate sparsity of encoded events
    ///
    /// **Deep Debt**: Runtime metric, no assumptions
    #[must_use]
    pub fn measure_sparsity(&self, input: &[f32]) -> f32 {
        if input.is_empty() {
            return 0.0;
        }

        let above_threshold = input.iter().filter(|&&val| val > self.threshold).count();

        1.0 - (above_threshold as f32 / input.len() as f32)
    }
}

impl Default for EventCodec {
    fn default() -> Self {
        Self::new(0.1) // Default from MNIST validation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_full() {
        let codec = EventCodec::new(0.1);

        // Test data with sparsity
        let input = vec![0.0, 0.5, 0.0, 1.0, 0.05, 0.8];

        // Encode with full format (index + value)
        let events = codec.encode(&input);

        // Decode
        let output = codec.decode(&events, input.len());

        // Check non-zero values preserved (with quantization)
        assert!((output[1] - 0.5).abs() < 0.01, "output[1] = {}", output[1]);
        assert!((output[3] - 1.0).abs() < 0.01, "output[3] = {}", output[3]);
        assert!((output[5] - 0.8).abs() < 0.01, "output[5] = {}", output[5]);

        // Check zeros preserved
        assert_eq!(output[0], 0.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[4], 0.0);
    }

    #[test]
    fn test_sparsity_measurement() {
        let codec = EventCodec::new(0.1);

        // 75% sparse
        let sparse_data = vec![0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0];
        let sparsity = codec.measure_sparsity(&sparse_data);
        assert!((sparsity - 0.75).abs() < 0.01);

        // 0% sparse (all above threshold)
        let dense_data = vec![0.5, 1.0, 0.8, 0.6];
        let sparsity = codec.measure_sparsity(&dense_data);
        assert!(sparsity < 0.01);
    }

    #[test]
    fn test_threshold_behavior() {
        // Low threshold: more events
        let low_codec = EventCodec::new(0.01);
        let data = vec![0.0, 0.02, 0.05, 0.1, 0.5];
        let low_events = low_codec.encode_simple(&data);

        // High threshold: fewer events
        let high_codec = EventCodec::new(0.2);
        let high_events = high_codec.encode_simple(&data);

        assert!(low_events.len() > high_events.len());
    }
}
