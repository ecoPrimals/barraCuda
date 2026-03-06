// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simple multi-layer perceptron with JSON weight serialization.
//!
//! Designed to replace ~400 LOC of hand-rolled MLP inference across
//! 3 WDM surrogates (ESN readout, SQW regressor, transport model).
//!
//! Provenance: neuralSpring `TOADSTOOL_HANDOFF` → toadStool absorption (S70).
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::nn::simple_mlp::SimpleMlp;
//!
//! let mlp = SimpleMlp::from_json(include_str!("weights.json"))?;
//! let output = mlp.forward(&input);
//! ```

use serde::{Deserialize, Serialize};

/// Activation function applied after each hidden layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    /// `ReLU`: max(0, x).
    Relu,
    /// Hyperbolic tangent.
    Tanh,
    /// Sigmoid: 1 / (1 + exp(-x)).
    Sigmoid,
    /// GELU approximation.
    Gelu,
    /// Identity (no activation).
    Identity,
}

/// A single dense layer: y = W·x + b
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Weight matrix (row-major: `out_size` × `in_size`).
    pub weight: Vec<Vec<f64>>,
    /// Bias vector (length `out_size`).
    pub bias: Vec<f64>,
    /// Activation applied after affine transform.
    pub activation: Activation,
}

/// Simple feed-forward MLP with JSON weight loading.
///
/// Stores layers as dense weight matrices (row-major).
/// Inference is pure CPU — for GPU inference, dispatch via Tensor ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMlp {
    /// Ordered dense layers.
    pub layers: Vec<DenseLayer>,
}

impl SimpleMlp {
    /// Construct from a JSON string containing serialized weights.
    /// # Errors
    /// Returns an error if JSON is malformed or shapes are inconsistent.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON string.
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create from explicit layer specifications.
    #[must_use]
    pub fn new(layers: Vec<DenseLayer>) -> Self {
        Self { layers }
    }

    /// Forward pass (CPU inference).
    /// Applies each layer in sequence: affine transform + activation.
    #[must_use]
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();

        for layer in &self.layers {
            let out_size = layer.bias.len();
            let mut y = Vec::with_capacity(out_size);

            for (row, b) in layer.weight.iter().zip(&layer.bias) {
                let dot: f64 = row.iter().zip(&x).map(|(w, xi)| w * xi).sum();
                y.push(dot + b);
            }

            apply_activation(&mut y, layer.activation);
            x = y;
        }

        x
    }

    /// Number of input features expected (from first layer).
    #[must_use]
    pub fn input_size(&self) -> Option<usize> {
        self.layers
            .first()
            .and_then(|l| l.weight.first().map(Vec::len))
    }

    /// Number of output features (from last layer).
    #[must_use]
    pub fn output_size(&self) -> Option<usize> {
        self.layers.last().map(|l| l.bias.len())
    }
}

fn apply_activation(values: &mut [f64], activation: Activation) {
    match activation {
        Activation::Relu => {
            for v in values.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Activation::Tanh => {
            for v in values.iter_mut() {
                *v = v.tanh();
            }
        }
        Activation::Sigmoid => {
            for v in values.iter_mut() {
                *v = if *v >= 0.0 {
                    1.0 / (1.0 + (-*v).exp())
                } else {
                    let ez = v.exp();
                    ez / (1.0 + ez)
                };
            }
        }
        Activation::Gelu => {
            for v in values.iter_mut() {
                let x = *v;
                *v = 0.5 * x * (1.0 + (0.797_884_560_8 * (x + 0.044_715 * x * x * x)).tanh());
            }
        }
        Activation::Identity => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mlp() -> SimpleMlp {
        SimpleMlp::new(vec![
            DenseLayer {
                weight: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
                bias: vec![0.0, 0.0, -0.5],
                activation: Activation::Relu,
            },
            DenseLayer {
                weight: vec![vec![1.0, 1.0, 1.0]],
                bias: vec![0.0],
                activation: Activation::Identity,
            },
        ])
    }

    #[test]
    fn test_forward_basic() {
        let mlp = make_test_mlp();
        let out = mlp.forward(&[1.0, 2.0]);
        // Layer 1: [1, 2, 2.5] after relu
        // Layer 2: [1+2+2.5] = [5.5]
        assert!((out[0] - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_json_roundtrip() {
        let mlp = make_test_mlp();
        let json = mlp.to_json().unwrap();
        let mlp2 = SimpleMlp::from_json(&json).unwrap();
        assert_eq!(mlp.layers.len(), mlp2.layers.len());
        let out1 = mlp.forward(&[3.0, -1.0]);
        let out2 = mlp2.forward(&[3.0, -1.0]);
        assert!((out1[0] - out2[0]).abs() < 1e-15);
    }

    #[test]
    fn test_input_output_size() {
        let mlp = make_test_mlp();
        assert_eq!(mlp.input_size(), Some(2));
        assert_eq!(mlp.output_size(), Some(1));
    }

    #[test]
    fn test_relu_negative() {
        let mlp = SimpleMlp::new(vec![DenseLayer {
            weight: vec![vec![1.0]],
            bias: vec![-10.0],
            activation: Activation::Relu,
        }]);
        let out = mlp.forward(&[5.0]);
        assert!(out[0].abs() < 1e-15);
    }

    #[test]
    fn test_sigmoid_activation() {
        let mlp = SimpleMlp::new(vec![DenseLayer {
            weight: vec![vec![1.0]],
            bias: vec![0.0],
            activation: Activation::Sigmoid,
        }]);
        let out = mlp.forward(&[0.0]);
        assert!((out[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_activation() {
        let mlp = SimpleMlp::new(vec![DenseLayer {
            weight: vec![vec![1.0]],
            bias: vec![0.0],
            activation: Activation::Tanh,
        }]);
        let out = mlp.forward(&[0.0]);
        assert!(out[0].abs() < 1e-10);
    }
}
