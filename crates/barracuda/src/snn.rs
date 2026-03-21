// SPDX-License-Identifier: AGPL-3.0-or-later
//! High-level Spiking Neural Network (SNN) API
//!
//! **EVOLVED**: Pure Rust spike processing - No specialized WGSL shaders!
//!
//! This module provides a production-ready interface for building and running
//! spiking neural networks using pure Rust event processing algorithms.
//!
//! # Spiking Neural Networks
//!
//! SNNs are brain-inspired neural networks that:
//! - Process information as discrete events (spikes)
//! - Maintain temporal dynamics (memory)
//! - Operate efficiently on any hardware
//! - Excel at temporal pattern recognition
//!
//! # Philosophy
//!
//! SNN operations are **event processing**, not heavy tensor math! For typical
//! neuron counts (100-10,000), pure Rust spike logic is faster and simpler
//! than GPU shader overhead.
//!
//! # Deep Debt Compliance
//!
//! - ✅ **Hardware agnostic**: No GPU/NPU assumptions
//! - ✅ **Pure Rust**: No specialized WGSL shaders
//! - ✅ **Fast**: Event processing beats GPU overhead
//! - ✅ **Safe**: Zero unsafe code
//! - ✅ **Capability-based**: Runtime configuration
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::snn::{SpikingNetwork, SNNConfig, SNNLayer};
//!
//! // No device needed - pure Rust!
//! let mut network = SpikingNetwork::builder()
//!     .add_layer(SNNLayer::LIF {
//!         size: 100,
//!         tau: 20.0,
//!         threshold: 1.0,
//!         reset: 0.0,
//!     })
//!     .add_layer(SNNLayer::TemporalPool { window_size: 10 })
//!     .build();
//!
//! // Process temporal sequence
//! let input_sequence = vec![/* spike trains */];
//! let output = network.process_sequence(&input_sequence)?;
//! ```

use crate::error::{BarracudaError, Result as BarracudaResult};
use rand::{Rng, SeedableRng};

/// Configuration for spiking neural network.
#[derive(Debug, Clone)]
pub struct SNNConfig {
    /// Input encoding parameters
    pub input_encoding: EncodingType,

    /// Output decoding parameters
    pub output_decoding: DecodingType,

    /// Enable automatic state management
    pub auto_reset: bool,

    /// Time step for simulation (ms)
    pub dt: f32,
}

impl Default for SNNConfig {
    fn default() -> Self {
        Self {
            input_encoding: EncodingType::Rate { max_rate: 100.0 },
            output_decoding: DecodingType::Rate,
            auto_reset: true,
            dt: 1.0,
        }
    }
}

/// Input encoding strategy (capability-based).
#[derive(Debug, Clone)]
pub enum EncodingType {
    /// Rate coding: intensity → spike frequency
    Rate {
        /// Maximum spike rate (Hz)
        max_rate: f32,
    },
    /// Temporal coding: value → spike timing
    Temporal,
    /// Population coding: distributed representation
    Population {
        /// Number of neurons in population
        n_neurons: usize,
    },
}

/// Output decoding strategy (capability-based).
#[derive(Debug, Clone)]
pub enum DecodingType {
    /// Rate decoding: spike frequency → intensity
    Rate,
    /// First-to-spike: winner-take-all
    FirstSpike,
    /// Population vector: weighted combination
    PopulationVector,
}

/// SNN layer types (all capability-based).
#[derive(Debug, Clone)]
pub enum SNNLayer {
    /// Leaky Integrate-and-Fire neurons
    LIF {
        /// Number of neurons in the layer
        size: usize,
        /// Membrane time constant (ms)
        tau: f32,
        /// Spike threshold
        threshold: f32,
        /// Reset potential after spike
        reset: f32,
    },
    /// Temporal pooling (aggregation over time)
    TemporalPool {
        /// Number of time steps to aggregate
        window_size: usize,
    },
    /// Sparse linear transformation
    SparseLinear {
        /// Input dimension
        input_size: usize,
        /// Output dimension
        output_size: usize,
        /// Fraction of non-zero weights (0.0–1.0)
        sparsity: f32,
        /// Weight matrix (row-major); `None` = auto-initialize
        weights: Option<Vec<f32>>,
    },
    /// Rate encoding layer
    RateEncoder {
        /// Maximum spike rate (Hz)
        max_rate: f32,
    },
    /// Rate decoding layer
    RateDecoder,
}

/// Network state (runtime-managed, no mocks)
struct LayerState {
    /// Neuron membrane potentials
    membrane: Vec<f32>,

    /// Current spike state
    spikes: Vec<f32>,

    /// Temporal pool buffer (if applicable)
    temporal_buffer: Vec<Vec<f32>>,

    /// Synapttic weights (for linear layers)
    weights: Option<Vec<f32>>,
}

impl LayerState {
    fn new(size: usize) -> Self {
        Self {
            membrane: vec![0.0; size],
            spikes: vec![0.0; size],
            temporal_buffer: Vec::new(),
            weights: None,
        }
    }
}

/// Spiking Neural Network
///
/// **Pure Rust implementation** - No GPU dependencies!
pub struct SpikingNetwork {
    config: SNNConfig,
    layers: Vec<SNNLayer>,
    states: Vec<LayerState>,
}

/// Builder for constructing SNNs.
pub struct SNNBuilder {
    layers: Vec<SNNLayer>,
    config: SNNConfig,
}

impl SNNBuilder {
    fn new(config: SNNConfig) -> Self {
        Self {
            layers: Vec::new(),
            config,
        }
    }

    /// Add a layer to the network.
    #[must_use]
    pub fn add_layer(mut self, layer: SNNLayer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Build the network, initializing layer states and weights.
    ///
    /// For [`SNNLayer::SparseLinear`] layers without explicit weights,
    /// auto-initializes sparse random weights based on the sparsity parameter.
    #[must_use]
    pub fn build(self) -> SpikingNetwork {
        let mut states = Vec::new();

        // Initialize states for each layer
        for layer in &self.layers {
            let size = match layer {
                SNNLayer::LIF { size, .. } => *size,
                SNNLayer::TemporalPool { .. } => 0, // Stateless passthrough
                SNNLayer::SparseLinear { output_size, .. } => *output_size,
                SNNLayer::RateEncoder { .. } => 0,
                SNNLayer::RateDecoder => 0,
            };

            let mut state = LayerState::new(size);

            // Initialize weights for linear layers
            if let SNNLayer::SparseLinear {
                input_size,
                output_size,
                sparsity,
                weights,
            } = layer
            {
                let w = if let Some(w) = weights {
                    w.clone()
                } else {
                    // Auto-initialize sparse random weights
                    let mut rng = rand::rngs::StdRng::from_os_rng();
                    let mut weights = vec![0.0; input_size * output_size];
                    for w in &mut weights {
                        if rng.random::<f32>() < *sparsity {
                            *w = rng.random_range(-1.0..1.0);
                        }
                    }
                    weights
                };
                state.weights = Some(w);
            }

            states.push(state);
        }

        SpikingNetwork {
            config: self.config,
            layers: self.layers,
            states,
        }
    }
}

impl SpikingNetwork {
    /// Create a network builder.
    #[must_use]
    pub fn builder() -> SNNBuilder {
        SNNBuilder::new(SNNConfig::default())
    }

    /// Create a network builder with config.
    #[must_use]
    pub fn builder_with_config(config: SNNConfig) -> SNNBuilder {
        SNNBuilder::new(config)
    }

    /// Process a single time step through the network
    ///
    /// **Pure Rust** - Fast event processing!
    ///
    /// # Arguments
    ///
    /// * `input` - Input spikes (0 or 1) or continuous values (for encoding layer)
    ///
    /// # Returns
    ///
    /// Output spikes or decoded values
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if any layer processes input with mismatched dimensions.
    pub fn process_step(&mut self, input: impl Into<Vec<f32>>) -> BarracudaResult<Vec<f32>> {
        let mut current = input.into();

        for i in 0..self.layers.len() {
            current = self.process_layer(i, &current)?;
        }

        Ok(current)
    }

    /// Process a sequence through the network
    ///
    /// # Arguments
    ///
    /// * `sequence` - Sequence of inputs over time
    ///
    /// # Returns
    ///
    /// Sequence of outputs
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if any layer processes input with mismatched dimensions.
    pub fn process_sequence(&mut self, sequence: &[Vec<f32>]) -> BarracudaResult<Vec<Vec<f32>>> {
        sequence
            .iter()
            .map(|input| self.process_step(input.as_slice()))
            .collect()
    }

    /// Process a single layer (pure Rust!)
    fn process_layer(&mut self, layer_idx: usize, input: &[f32]) -> BarracudaResult<Vec<f32>> {
        let layer = &self.layers[layer_idx];
        let state = &mut self.states[layer_idx];

        match layer {
            SNNLayer::LIF {
                size,
                tau,
                threshold,
                reset,
            } => {
                if input.len() != *size {
                    return Err(BarracudaError::InvalidInput {
                        message: format!(
                            "LIF input size mismatch: expected {}, got {}",
                            size,
                            input.len()
                        ),
                    });
                }

                // Leaky Integrate-and-Fire dynamics (pure Rust!)
                let dt = self.config.dt;
                let decay_factor = 1.0 - (dt / tau);

                for i in 0..*size {
                    // Decay membrane potential
                    state.membrane[i] *= decay_factor;

                    // Integrate input
                    state.membrane[i] += input[i];

                    // Check for spike
                    if state.membrane[i] >= *threshold {
                        state.spikes[i] = 1.0;
                        state.membrane[i] = *reset; // Reset to reset potential
                    } else {
                        state.spikes[i] = 0.0;
                    }
                }

                Ok(state.spikes.clone())
            }

            SNNLayer::TemporalPool { window_size } => {
                // Temporal pooling: sum spikes over time window
                // Copy necessary: we must store the frame for later aggregation; buffer owns the data.
                state.temporal_buffer.push(input.to_vec());

                if state.temporal_buffer.len() > *window_size {
                    state.temporal_buffer.remove(0);
                }

                // Sum over time window
                let output_size = input.len();
                let mut pooled = vec![0.0; output_size];

                for frame in &state.temporal_buffer {
                    for (i, &spike) in frame.iter().enumerate() {
                        pooled[i] += spike;
                    }
                }

                Ok(pooled)
            }

            SNNLayer::SparseLinear {
                input_size,
                output_size,
                ..
            } => {
                if input.len() != *input_size {
                    return Err(BarracudaError::InvalidInput {
                        message: format!(
                            "Linear input size mismatch: expected {}, got {}",
                            input_size,
                            input.len()
                        ),
                    });
                }

                let weights = state.weights.as_ref().ok_or_else(|| {
                    crate::error::BarracudaError::InvalidOperation {
                        op: "SNN Dense layer".to_string(),
                        reason: "Dense layer weights not initialized".to_string(),
                    }
                })?;
                let mut output = vec![0.0; *output_size];

                // Matrix multiply (sparse)
                for i in 0..*output_size {
                    for j in 0..*input_size {
                        output[i] += weights[i * input_size + j] * input[j];
                    }
                }

                Ok(output)
            }

            SNNLayer::RateEncoder { max_rate } => {
                // Rate encoding: continuous → spikes
                let mut rng = rand::rngs::StdRng::from_os_rng();
                let dt = self.config.dt;

                let spikes: Vec<f32> = input
                    .iter()
                    .map(|&value| {
                        let spike_prob = value.abs() * max_rate * dt / 1000.0; // Convert to probability
                        if rng.random::<f32>() < spike_prob {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();

                Ok(spikes)
            }

            SNNLayer::RateDecoder => {
                // Rate decoding: spikes → continuous
                // Just pass through (decoding happens via temporal pooling)
                // Copy necessary: we must return owned Vec; input is borrowed.
                Ok(input.to_vec())
            }
        }
    }

    /// Reset all layer states.
    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.membrane.fill(0.0);
            state.spikes.fill(0.0);
            state.temporal_buffer.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snn_builder() {
        let network = SpikingNetwork::builder()
            .add_layer(SNNLayer::LIF {
                size: 10,
                tau: 20.0,
                threshold: 1.0,
                reset: 0.0,
            })
            .build();

        assert_eq!(network.layers.len(), 1);
    }

    #[test]
    fn test_lif_spike() {
        let mut network = SpikingNetwork::builder()
            .add_layer(SNNLayer::LIF {
                size: 5,
                tau: 10.0,
                threshold: 1.0,
                reset: 0.0,
            })
            .build();

        // Strong input should cause spikes
        let input = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let output = network.process_step(input.as_slice()).unwrap();

        // Should have spikes (1.0) since input > threshold
        assert!(output.contains(&1.0));
    }

    #[test]
    fn test_lif_no_spike() {
        let mut network = SpikingNetwork::builder()
            .add_layer(SNNLayer::LIF {
                size: 5,
                tau: 10.0,
                threshold: 1.0,
                reset: 0.0,
            })
            .build();

        // Weak input should not cause spikes
        let input = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let output = network.process_step(input.as_slice()).unwrap();

        // Should have no spikes
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_temporal_pool() {
        let mut network = SpikingNetwork::builder()
            .add_layer(SNNLayer::TemporalPool { window_size: 3 })
            .build();

        // Process sequence
        network.process_step([1.0, 0.0]).unwrap();
        network.process_step([0.0, 1.0]).unwrap();
        let output = network.process_step([1.0, 0.0]).unwrap();

        // Should sum over last 3 frames
        assert_eq!(output[0], 2.0); // Two 1.0s in first channel
        assert_eq!(output[1], 1.0); // One 1.0 in second channel
    }

    #[test]
    fn test_sparse_linear() {
        let weights = vec![
            0.5, 0.0, 0.0, 0.5, // Output 0 weights
            0.0, 0.5, 0.5, 0.0, // Output 1 weights
        ];

        let mut network = SpikingNetwork::builder()
            .add_layer(SNNLayer::SparseLinear {
                input_size: 4,
                output_size: 2,
                sparsity: 0.5,
                weights: Some(weights),
            })
            .build();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = network.process_step(input.as_slice()).unwrap();

        // output[0] = 0.5*1.0 + 0.5*4.0 = 2.5
        // output[1] = 0.5*2.0 + 0.5*3.0 = 2.5
        assert!((output[0] - 2.5).abs() < 1e-6);
        assert!((output[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_rate_encoding() {
        let mut network = SpikingNetwork::builder()
            .add_layer(SNNLayer::RateEncoder { max_rate: 100.0 })
            .build();

        // High values should produce more spikes (statistically)
        let high_input = vec![1.0; 10];
        let low_input = vec![0.1; 10];

        let mut high_spike_count = 0;
        let mut low_spike_count = 0;

        // Run multiple trials
        for _ in 0..1000 {
            let high_out = network.process_step(high_input.as_slice()).unwrap();
            let low_out = network.process_step(low_input.as_slice()).unwrap();

            high_spike_count += high_out.iter().filter(|&&x| x == 1.0).count();
            low_spike_count += low_out.iter().filter(|&&x| x == 1.0).count();
        }

        // High input should produce more spikes
        assert!(high_spike_count > low_spike_count * 2);
    }

    #[test]
    fn test_network_reset() {
        let mut network = SpikingNetwork::builder()
            .add_layer(SNNLayer::LIF {
                size: 5,
                tau: 10.0,
                threshold: 1.0,
                reset: 0.0,
            })
            .build();

        // Process some input
        network.process_step([0.5; 5]).unwrap();

        // Reset
        network.reset();

        // State should be zero
        assert!(network.states[0].membrane.iter().all(|&x| x == 0.0));
        assert!(network.states[0].spikes.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_multi_layer_network() {
        let mut network = SpikingNetwork::builder()
            .add_layer(SNNLayer::LIF {
                size: 10,
                tau: 20.0,
                threshold: 1.0,
                reset: 0.0,
            })
            .add_layer(SNNLayer::TemporalPool { window_size: 5 })
            .build();

        // Process sequence
        let sequence = vec![vec![1.5; 10], vec![1.5; 10], vec![0.0; 10]];

        let outputs = network.process_sequence(&sequence).unwrap();
        assert_eq!(outputs.len(), 3);
    }
}
