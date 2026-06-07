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

use rand::Rng;
use serde::{Deserialize, Serialize};

mod serialization;

pub use serialization::ModelBinaryError;

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

    /// Train the MLP using mini-batch SGD with backpropagation.
    ///
    /// # Arguments
    /// * `inputs` — training input vectors
    /// * `targets` — target output vectors (must match `inputs` length)
    /// * `config` — training hyperparameters
    ///
    /// # Returns
    /// Mean squared error on the final epoch.
    ///
    /// # Errors
    /// Returns an error if inputs/targets are empty, lengths mismatch,
    /// or dimensions are inconsistent with the network architecture.
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        config: &TrainConfig,
    ) -> Result<f64, TrainError> {
        if inputs.is_empty() {
            return Err(TrainError::EmptyData);
        }
        if inputs.len() != targets.len() {
            return Err(TrainError::LengthMismatch {
                inputs: inputs.len(),
                targets: targets.len(),
            });
        }
        if let Some(expected_in) = self.input_size() {
            if inputs.iter().any(|x| x.len() != expected_in) {
                return Err(TrainError::DimensionMismatch {
                    expected: expected_in,
                    context: "input",
                });
            }
        }
        if let Some(expected_out) = self.output_size() {
            if targets.iter().any(|t| t.len() != expected_out) {
                return Err(TrainError::DimensionMismatch {
                    expected: expected_out,
                    context: "target",
                });
            }
        }

        let n = inputs.len();
        let mut final_mse = 0.0;

        for _epoch in 0..config.epochs {
            let mut epoch_loss = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass — store pre-activation and post-activation for each layer.
                let (activations, pre_activations) = self.forward_with_cache(input);

                // Backward pass — compute gradients.
                let n_layers = self.layers.len();
                let mut delta = Vec::with_capacity(target.len());
                let output = &activations[n_layers];
                for (o, t) in output.iter().zip(target.iter()) {
                    delta.push(o - t);
                    epoch_loss += (o - t).powi(2);
                }

                for l in (0..n_layers).rev() {
                    let act_deriv =
                        activation_derivative(&pre_activations[l], self.layers[l].activation);
                    let grad: Vec<f64> = delta.iter().zip(&act_deriv).map(|(d, a)| d * a).collect();

                    let prev_act = &activations[l];
                    let layer = &mut self.layers[l];

                    for (i, g) in grad.iter().enumerate() {
                        layer.bias[i] -= config.learning_rate * g;
                        for (j, p) in prev_act.iter().enumerate() {
                            layer.weight[i][j] -= config.learning_rate * g * p;
                        }
                    }

                    if l > 0 {
                        let mut new_delta = vec![0.0; self.layers[l].weight[0].len()];
                        for (i, g) in grad.iter().enumerate() {
                            for (j, nd) in new_delta.iter_mut().enumerate() {
                                *nd += g * self.layers[l].weight[i][j];
                            }
                        }
                        delta = new_delta;
                    }
                }
            }

            final_mse = epoch_loss / (n as f64 * self.output_size().unwrap_or(1) as f64);
        }

        Ok(final_mse)
    }

    /// Forward pass that caches intermediate activations for backprop.
    fn forward_with_cache(&self, input: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        activations.push(input.to_vec());

        let mut x = input.to_vec();
        for layer in &self.layers {
            let out_size = layer.bias.len();
            let mut pre_act = Vec::with_capacity(out_size);
            for (row, b) in layer.weight.iter().zip(&layer.bias) {
                let dot: f64 = row.iter().zip(&x).map(|(w, xi)| w * xi).sum();
                pre_act.push(dot + b);
            }
            let mut post_act = pre_act.clone();
            apply_activation(&mut post_act, layer.activation);
            pre_activations.push(pre_act);
            activations.push(post_act.clone());
            x = post_act;
        }

        (activations, pre_activations)
    }
}

/// Training hyperparameters for `SimpleMlp::train`.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Learning rate (step size).
    pub learning_rate: f64,
    /// Number of full passes over the dataset.
    pub epochs: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 100,
        }
    }
}

/// Errors that can occur during MLP training.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    /// No training samples provided.
    EmptyData,
    /// Inputs and targets have different counts.
    LengthMismatch {
        /// Number of input samples.
        inputs: usize,
        /// Number of target samples.
        targets: usize,
    },
    /// A sample has wrong dimensionality.
    DimensionMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Whether this is "input" or "target".
        context: &'static str,
    },
}

impl std::fmt::Display for TrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "training data is empty"),
            Self::LengthMismatch { inputs, targets } => {
                write!(
                    f,
                    "inputs ({inputs}) and targets ({targets}) length mismatch"
                )
            }
            Self::DimensionMismatch { expected, context } => {
                write!(f, "{context} dimension mismatch (expected {expected})")
            }
        }
    }
}

impl std::error::Error for TrainError {}

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
        assert!(mse < 0.1, "MSE should decrease with training, got {mse}");
    }

    #[test]
    fn test_train_linear() {
        let mut mlp = SimpleMlp::new(vec![DenseLayer {
            weight: vec![vec![0.1]],
            bias: vec![0.0],
            activation: Activation::Identity,
        }]);

        let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let targets: Vec<Vec<f64>> = inputs
            .iter()
            .map(|x| vec![x[0].mul_add(2.0, 1.0)])
            .collect();

        let config = TrainConfig {
            learning_rate: 0.0001,
            epochs: 200,
        };
        let mse = mlp.train(&inputs, &targets, &config).unwrap();
        assert!(mse < 5.0, "linear fit should converge, got MSE={mse}");
    }

    #[test]
    fn test_train_errors() {
        let mut mlp = SimpleMlp::new(vec![DenseLayer {
            weight: vec![vec![1.0]],
            bias: vec![0.0],
            activation: Activation::Relu,
        }]);
        let config = TrainConfig::default();

        assert_eq!(
            mlp.train(&[], &[], &config).unwrap_err(),
            TrainError::EmptyData
        );
        assert_eq!(
            mlp.train(&[vec![1.0]], &[], &config).unwrap_err(),
            TrainError::LengthMismatch {
                inputs: 1,
                targets: 0
            }
        );
        assert_eq!(
            mlp.train(&[vec![1.0, 2.0]], &[vec![1.0]], &config)
                .unwrap_err(),
            TrainError::DimensionMismatch {
                expected: 1,
                context: "input"
            }
        );
    }

    #[test]
    fn test_from_dims_single_layer() {
        let mlp = SimpleMlp::from_dims(&[36, 16], Activation::Sigmoid);
        assert_eq!(mlp.layers.len(), 1);
        assert_eq!(mlp.input_size(), Some(36));
        assert_eq!(mlp.output_size(), Some(16));
        assert_eq!(mlp.layers[0].activation, Activation::Identity);
    }

    #[test]
    fn test_from_dims_multi_layer() {
        let mlp = SimpleMlp::from_dims(&[36, 64, 16], Activation::Relu);
        assert_eq!(mlp.layers.len(), 2);
        assert_eq!(mlp.input_size(), Some(36));
        assert_eq!(mlp.output_size(), Some(16));
        assert_eq!(mlp.layers[0].activation, Activation::Relu);
        assert_eq!(mlp.layers[1].activation, Activation::Identity);
    }

    #[test]
    fn test_from_dims_trains_successfully() {
        let mut mlp = SimpleMlp::from_dims(&[4, 2], Activation::Sigmoid);
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let targets = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];
        let config = TrainConfig {
            learning_rate: 0.1,
            epochs: 100,
        };
        let mse = mlp.train(&inputs, &targets, &config).unwrap();
        assert!(mse < 0.25, "from_dims should train, got MSE={mse}");
    }
}
