// SPDX-License-Identifier: AGPL-3.0-or-later
//! Event Codec - Dense ↔ Sparse Event Conversion
//!
//! Converts between dense tensor representations and sparse event streams
//! for NPU execution. This is where the energy efficiency comes from!
//!
//! **Deep Debt Principles**:
//! - Pure Rust conversion (no unsafe)
//! - Runtime threshold configuration
//! - No hardcoded encoding schemes

/// Event codec for NPU dense/sparse conversion
pub struct EventCodec {
    /// Threshold for event generation (0.0-1.0)
    threshold: f32,
}

impl EventCodec {
    /// Create new codec with threshold
    ///
    /// **Deep Debt**: Configurable threshold, no hardcoding
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Convert dense activations to sparse events
    ///
    /// Only encodes values above threshold. This is where NPU's
    /// energy efficiency comes from - sparse event processing!
    ///
    /// **Deep Debt**: Pure Rust, no unsafe
    ///
    /// # Arguments
    /// * `input` - Dense tensor data
    ///
    /// # Returns
    /// Sparse event stream (u8 encoded)
    pub fn encode(&self, input: &[f32]) -> Vec<u8> {
        input
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > self.threshold)
            .flat_map(|(idx, &val)| {
                // Encode as: index (u32 LE) + value (u8)
                let mut bytes = Vec::with_capacity(5);
                bytes.extend_from_slice(&(idx as u32).to_le_bytes());
                bytes.push((val * 255.0).clamp(0.0, 255.0) as u8);
                bytes
            })
            .collect()
    }

    /// Convert dense to events (simplified for small inputs)
    ///
    /// For small inputs (like MNIST 784 dims), use simpler encoding
    ///
    /// **Deep Debt**: No unsafe, runtime size adaptation
    pub fn encode_simple(&self, input: &[f32]) -> Vec<u8> {
        input
            .iter()
            .filter(|&&val| val > self.threshold)
            .map(|&val| (val * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
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

#[expect(clippy::unwrap_used, reason = "tests")]
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
