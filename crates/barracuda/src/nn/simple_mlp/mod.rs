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

mod serialization;
mod training;

pub use serialization::ModelBinaryError;
pub use training::{TrainConfig, TrainError};

use rand::Rng;
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
    #[serde(alias = "weights")]
    pub weight: Vec<Vec<f64>>,
    /// Bias vector (length `out_size`).
    #[serde(alias = "biases")]
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

    /// Create from a list of layer dimensions with Xavier-uniform random init.
    ///
    /// `dims` specifies the size of each layer including input:
    /// e.g. `&[36, 16]` creates one layer mapping 36 inputs → 16 outputs.
    /// `&[36, 64, 16]` creates two layers: 36→64 and 64→16.
    ///
    /// All hidden layers use `activation`; the last layer uses `Identity`
    /// (softmax/sigmoid is applied externally by the consumer).
    #[must_use]
    pub fn from_dims(dims: &[usize], activation: Activation) -> Self {
        let mut rng = rand::rng();
        let mut layers = Vec::with_capacity(dims.len().saturating_sub(1));

        for i in 0..dims.len().saturating_sub(1) {
            let in_size = dims[i];
            let out_size = dims[i + 1];
            let limit = (6.0_f64 / (in_size + out_size) as f64).sqrt();

            let weight: Vec<Vec<f64>> = (0..out_size)
                .map(|_| {
                    (0..in_size)
                        .map(|_| rng.random_range(-limit..limit))
                        .collect()
                })
                .collect();
            let bias = vec![0.0; out_size];

            let act = if i + 1 < dims.len().saturating_sub(1) {
                activation
            } else {
                Activation::Identity
            };

            layers.push(DenseLayer {
                weight,
                bias,
                activation: act,
            });
        }

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

/// Compute the derivative of an activation function given pre-activation values.
fn activation_derivative(pre_act: &[f64], activation: Activation) -> Vec<f64> {
    match activation {
        Activation::Relu => pre_act
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect(),
        Activation::Tanh => pre_act
            .iter()
            .map(|&x| {
                let t = x.tanh();
                (-t).mul_add(t, 1.0)
            })
            .collect(),
        Activation::Sigmoid => pre_act
            .iter()
            .map(|&x| {
                let s = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let ez = x.exp();
                    ez / (1.0 + ez)
                };
                s * (1.0 - s)
            })
            .collect(),
        Activation::Gelu => pre_act
            .iter()
            .map(|&x| {
                let c = 0.797_884_560_8;
                let inner = c * 0.044_715f64.mul_add(x * x * x, x);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;
                let d_inner = c * 0.044_715f64.mul_add(3.0 * x * x, 1.0);
                0.5f64.mul_add(1.0 + tanh_val, 0.5 * x * sech2 * d_inner)
            })
            .collect(),
        Activation::Identity => vec![1.0; pre_act.len()],
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
                *v = 0.5 * x * (1.0 + (0.797_884_560_8 * (0.044_715 * x * x).mul_add(x, x)).tanh());
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

    #[test]
    fn test_train_xor() {
        let mut mlp = SimpleMlp::new(vec![
            DenseLayer {
                weight: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                bias: vec![0.0, -0.5],
                activation: Activation::Tanh,
            },
            DenseLayer {
                weight: vec![vec![0.5, -0.5]],
                bias: vec![0.0],
                activation: Activation::Identity,
            },
        ]);

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let config = TrainConfig {
            learning_rate: 0.1,
            epochs: 500,
        };
        let mse = mlp.train(&inputs, &targets, &config).unwrap();
        assert!(mse < 0.1, "XOR training did not converge, MSE: {mse}");
    }

    #[test]
    fn test_train_empty_data() {
        let mut mlp = make_test_mlp();
        let err = mlp.train(&[], &[], &TrainConfig::default()).unwrap_err();
        assert_eq!(err, TrainError::EmptyData);
    }

    #[test]
    fn test_train_length_mismatch() {
        let mut mlp = make_test_mlp();
        let err = mlp
            .train(
                &[vec![1.0, 2.0]],
                &[vec![1.0], vec![2.0]],
                &TrainConfig::default(),
            )
            .unwrap_err();
        assert!(matches!(err, TrainError::LengthMismatch { .. }));
    }

    #[test]
    fn test_from_dims_shapes() {
        let mlp = SimpleMlp::from_dims(&[36, 64, 16], Activation::Relu);
        assert_eq!(mlp.layers.len(), 2);
        assert_eq!(mlp.input_size(), Some(36));
        assert_eq!(mlp.output_size(), Some(16));
        assert_eq!(mlp.layers[0].activation, Activation::Relu);
        assert_eq!(mlp.layers[1].activation, Activation::Identity);
    }

    #[test]
    fn test_gelu_activation() {
        let mlp = SimpleMlp::new(vec![DenseLayer {
            weight: vec![vec![1.0]],
            bias: vec![0.0],
            activation: Activation::Gelu,
        }]);
        let out = mlp.forward(&[0.0]);
        assert!(out[0].abs() < 1e-10);
        let out2 = mlp.forward(&[1.0]);
        assert!(out2[0] > 0.8);
    }

    #[test]
    fn test_identity_activation() {
        let mlp = SimpleMlp::new(vec![DenseLayer {
            weight: vec![vec![2.0]],
            bias: vec![3.0],
            activation: Activation::Identity,
        }]);
        let out = mlp.forward(&[5.0]);
        assert!((out[0] - 13.0).abs() < 1e-10);
    }

    #[test]
    fn binary_roundtrip_single_layer() {
        let mlp = SimpleMlp::from_dims(&[4, 2], Activation::Relu);
        let data = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_binary(&data).unwrap();
        assert_eq!(mlp.layers.len(), restored.layers.len());
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let diff: f64 = mlp
            .forward(&input)
            .iter()
            .zip(restored.forward(&input).iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-15);
    }

    #[test]
    fn binary_roundtrip_multi_layer() {
        let mlp = SimpleMlp::from_dims(&[8, 16, 4], Activation::Tanh);
        let data = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_binary(&data).unwrap();
        assert_eq!(mlp.layers.len(), restored.layers.len());
    }

    #[test]
    fn binary_smaller_than_json() {
        let mlp = SimpleMlp::from_dims(&[36, 16], Activation::Relu);
        let bin = mlp.to_binary().unwrap();
        let json = mlp.to_json().unwrap();
        assert!(
            bin.len() < json.len(),
            "binary ({}) should be smaller than JSON ({})",
            bin.len(),
            json.len()
        );
    }

    #[test]
    fn binary_header_magic_and_version() {
        let mlp = SimpleMlp::from_dims(&[2, 1], Activation::Relu);
        let data = mlp.to_binary().unwrap();
        assert_eq!(&data[0..4], b"BCML");
        assert_eq!(data[4], 1); // version
        assert_eq!(data[5], 1); // format (bincode)
    }

    #[test]
    fn binary_checksum_verification_fails_on_tamper() {
        let mlp = SimpleMlp::from_dims(&[4, 2], Activation::Relu);
        let mut data = mlp.to_binary().unwrap();
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        let err = SimpleMlp::from_binary(&data).unwrap_err();
        assert!(
            matches!(err, ModelBinaryError::ChecksumMismatch),
            "expected ChecksumMismatch, got: {err:?}"
        );
    }

    #[test]
    fn binary_too_short_rejected() {
        let err = SimpleMlp::from_binary(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, ModelBinaryError::TooShort(_)));
    }

    #[test]
    fn binary_bad_magic_rejected() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"NOPE");
        let err = SimpleMlp::from_binary(&data).unwrap_err();
        assert!(matches!(err, ModelBinaryError::BadMagic));
    }

    #[test]
    fn binary_unsupported_version_rejected() {
        let mlp = SimpleMlp::from_dims(&[2, 1], Activation::Relu);
        let mut data = mlp.to_binary().unwrap();
        data[4] = 99;
        let err = SimpleMlp::from_binary(&data).unwrap_err();
        assert!(matches!(err, ModelBinaryError::UnsupportedVersion(99)));
    }

    #[test]
    fn from_auto_detects_binary() {
        let mlp = SimpleMlp::from_dims(&[4, 2], Activation::Tanh);
        let bin = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_auto(&bin).unwrap();
        assert_eq!(restored.layers.len(), mlp.layers.len());
    }

    #[test]
    fn from_auto_detects_json() {
        let mlp = SimpleMlp::from_dims(&[8, 4], Activation::Tanh);
        let json = mlp.to_json().unwrap();
        let restored = SimpleMlp::from_auto(json.as_bytes()).unwrap();
        for (orig, rest) in mlp.layers[0]
            .weight
            .iter()
            .flatten()
            .zip(restored.layers[0].weight.iter().flatten())
        {
            assert!((orig - rest).abs() < 1e-14, "JSON roundtrip precision loss");
        }
    }

    #[test]
    fn serde_alias_weights_biases() {
        let json = r#"{
            "layers": [{
                "weights": [[1.0, 2.0], [3.0, 4.0]],
                "biases": [0.1, 0.2],
                "activation": "relu"
            }]
        }"#;
        let mlp = SimpleMlp::from_json(json).unwrap();
        assert_eq!(mlp.layers[0].weight, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(mlp.layers[0].bias, vec![0.1, 0.2]);
    }

    #[test]
    fn serde_canonical_weight_bias() {
        let json = r#"{
            "layers": [{
                "weight": [[5.0, 6.0]],
                "bias": [0.5],
                "activation": "sigmoid"
            }]
        }"#;
        let mlp = SimpleMlp::from_json(json).unwrap();
        assert_eq!(mlp.layers[0].weight, vec![vec![5.0, 6.0]]);
        assert_eq!(mlp.layers[0].bias, vec![0.5]);
    }

    #[test]
    fn binary_trained_model_inference_matches() {
        let mut mlp = SimpleMlp::from_dims(&[4, 8, 2], Activation::Tanh);
        let inputs = vec![vec![1.0, 2.0, 3.0, 4.0]; 10];
        let targets = vec![vec![0.5, -0.5]; 10];
        let config = TrainConfig {
            learning_rate: 0.01,
            epochs: 50,
        };
        mlp.train(&inputs, &targets, &config).unwrap();

        let bin_data = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_binary(&bin_data).unwrap();

        let test_input = vec![2.0, 3.0, 1.0, 0.5];
        let orig_out = mlp.forward(&test_input);
        let rest_out = restored.forward(&test_input);
        for (a, b) in orig_out.iter().zip(rest_out.iter()) {
            assert!((a - b).abs() < 1e-15, "binary roundtrip changed inference");
        }
    }
}
