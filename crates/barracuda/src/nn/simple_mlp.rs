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

/// Magic bytes identifying a barraCuda ML binary model file.
const MODEL_MAGIC: &[u8; 4] = b"BCML";
/// Header version (1 = initial release).
const MODEL_VERSION: u8 = 1;
/// Format tag for bincode-encoded payload.
const FORMAT_BINCODE: u8 = 1;
/// Total header size: magic(4) + version(1) + format(1) + reserved(2) + len(4) + blake3(32).
const MODEL_HEADER_SIZE: usize = 44;

/// Errors from binary model serialization/deserialization.
#[derive(Debug, thiserror::Error)]
pub enum ModelBinaryError {
    /// File too short to contain a valid header.
    #[error("data too short ({0} bytes, need at least {MODEL_HEADER_SIZE})")]
    TooShort(usize),
    /// File does not start with `BCML` magic.
    #[error("invalid magic bytes (expected BCML header)")]
    BadMagic,
    /// Unsupported header version.
    #[error("unsupported model version {0} (expected {MODEL_VERSION})")]
    UnsupportedVersion(u8),
    /// Unsupported format tag.
    #[error("unsupported format tag {0}")]
    UnsupportedFormat(u8),
    /// BLAKE3 checksum mismatch — data corrupted or tampered.
    #[error("BLAKE3 checksum mismatch (data integrity failure)")]
    ChecksumMismatch,
    /// Payload exceeds u32 address space.
    #[error("payload too large ({0} bytes, max 4GB)")]
    PayloadTooLarge(usize),
    /// Encoding failed.
    #[error("encode error: {0}")]
    Encode(String),
    /// Decoding failed.
    #[error("decode error: {0}")]
    Decode(String),
}

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

    /// Serialize to binary format with BLAKE3 integrity header.
    ///
    /// File layout (44-byte header + payload):
    /// - Magic: `BCML` (4 bytes)
    /// - Version: u8 (1 = initial)
    /// - Format: u8 (1 = bincode)
    /// - Reserved: 2 bytes
    /// - Payload length: u32 LE
    /// - BLAKE3 checksum of payload: 32 bytes
    /// - Payload (bincode-encoded `SimpleMlp`)
    /// # Errors
    /// Returns an error if bincode serialization fails.
    pub fn to_binary(&self) -> Result<Vec<u8>, ModelBinaryError> {
        let payload = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| ModelBinaryError::Encode(e.to_string()))?;

        let checksum = blake3::hash(&payload);
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| ModelBinaryError::PayloadTooLarge(payload.len()))?;

        let mut buf = Vec::with_capacity(MODEL_HEADER_SIZE + payload.len());
        buf.extend_from_slice(MODEL_MAGIC);
        buf.push(MODEL_VERSION);
        buf.push(FORMAT_BINCODE);
        buf.extend_from_slice(&[0u8; 2]);
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(checksum.as_bytes());
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    /// Deserialize from binary format, verifying BLAKE3 integrity.
    /// # Errors
    /// Returns an error if the header is invalid, checksum fails, or
    /// bincode deserialization fails.
    pub fn from_binary(data: &[u8]) -> Result<Self, ModelBinaryError> {
        if data.len() < MODEL_HEADER_SIZE {
            return Err(ModelBinaryError::TooShort(data.len()));
        }
        if &data[0..4] != MODEL_MAGIC {
            return Err(ModelBinaryError::BadMagic);
        }
        let version = data[4];
        if version != MODEL_VERSION {
            return Err(ModelBinaryError::UnsupportedVersion(version));
        }
        let format = data[5];
        if format != FORMAT_BINCODE {
            return Err(ModelBinaryError::UnsupportedFormat(format));
        }

        let payload_len =
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let expected_total = MODEL_HEADER_SIZE + payload_len;
        if data.len() < expected_total {
            return Err(ModelBinaryError::TooShort(data.len()));
        }

        let stored_checksum: [u8; 32] = data[12..44].try_into().unwrap_or([0u8; 32]);
        let payload = &data[MODEL_HEADER_SIZE..expected_total];
        let computed = blake3::hash(payload);
        if computed.as_bytes() != &stored_checksum {
            return Err(ModelBinaryError::ChecksumMismatch);
        }

        let (mlp, _) =
            bincode::serde::decode_from_slice::<Self, _>(payload, bincode::config::standard())
                .map_err(|e| ModelBinaryError::Decode(e.to_string()))?;
        Ok(mlp)
    }

    /// Detect format from file bytes and deserialize accordingly.
    ///
    /// If the file starts with the `BCML` magic header, deserializes as binary.
    /// Otherwise falls back to JSON parsing.
    /// # Errors
    /// Returns an error if neither format succeeds.
    pub fn from_auto(data: &[u8]) -> Result<Self, ModelBinaryError> {
        if data.len() >= 4 && &data[0..4] == MODEL_MAGIC {
            Self::from_binary(data)
        } else {
            let json_str = std::str::from_utf8(data)
                .map_err(|e| ModelBinaryError::Decode(e.to_string()))?;
            Self::from_json(json_str)
                .map_err(|e| ModelBinaryError::Decode(e.to_string()))
        }
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
        let targets: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0].mul_add(2.0, 1.0)]).collect();

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

    // ── Binary serialization tests ──────────────────────────────────────

    #[test]
    fn binary_roundtrip_single_layer() {
        let mlp = SimpleMlp::from_dims(&[36, 16], Activation::Sigmoid);
        let bytes = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_binary(&bytes).unwrap();
        assert_eq!(restored.layers.len(), mlp.layers.len());
        assert_eq!(restored.layers[0].weight, mlp.layers[0].weight);
        assert_eq!(restored.layers[0].bias, mlp.layers[0].bias);
    }

    #[test]
    fn binary_roundtrip_multi_layer() {
        let mlp = SimpleMlp::from_dims(&[36, 64, 32, 16], Activation::Relu);
        let bytes = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_binary(&bytes).unwrap();
        assert_eq!(restored.layers.len(), 3);
        for (orig, rest) in mlp.layers.iter().zip(restored.layers.iter()) {
            assert_eq!(orig.weight, rest.weight);
            assert_eq!(orig.bias, rest.bias);
            assert_eq!(orig.activation, rest.activation);
        }
    }

    #[test]
    fn binary_smaller_than_json() {
        let mlp = SimpleMlp::from_dims(&[36, 16], Activation::Sigmoid);
        let bin_bytes = mlp.to_binary().unwrap();
        let json_bytes = mlp.to_json().unwrap().into_bytes();
        assert!(
            bin_bytes.len() < json_bytes.len(),
            "binary ({}) should be smaller than JSON ({})",
            bin_bytes.len(),
            json_bytes.len()
        );
    }

    #[test]
    fn binary_header_magic_and_version() {
        let mlp = SimpleMlp::from_dims(&[4, 2], Activation::Sigmoid);
        let bytes = mlp.to_binary().unwrap();
        assert_eq!(&bytes[0..4], b"BCML");
        assert_eq!(bytes[4], 1); // version
        assert_eq!(bytes[5], 1); // format = bincode
    }

    #[test]
    fn binary_checksum_verification_fails_on_tamper() {
        let mlp = SimpleMlp::from_dims(&[4, 2], Activation::Sigmoid);
        let mut bytes = mlp.to_binary().unwrap();
        let payload_start = 44;
        if bytes.len() > payload_start + 5 {
            bytes[payload_start + 5] ^= 0xFF;
        }
        let result = SimpleMlp::from_binary(&bytes);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ModelBinaryError::ChecksumMismatch),
            "tampered payload should fail checksum"
        );
    }

    #[test]
    fn binary_too_short_rejected() {
        let result = SimpleMlp::from_binary(&[0u8; 10]);
        assert!(matches!(result.unwrap_err(), ModelBinaryError::TooShort(10)));
    }

    #[test]
    fn binary_bad_magic_rejected() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(b"XXXX");
        let result = SimpleMlp::from_binary(&data);
        assert!(matches!(result.unwrap_err(), ModelBinaryError::BadMagic));
    }

    #[test]
    fn binary_unsupported_version_rejected() {
        let mlp = SimpleMlp::from_dims(&[4, 2], Activation::Sigmoid);
        let mut bytes = mlp.to_binary().unwrap();
        bytes[4] = 99; // bad version
        let result = SimpleMlp::from_binary(&bytes);
        assert!(matches!(
            result.unwrap_err(),
            ModelBinaryError::UnsupportedVersion(99)
        ));
    }

    #[test]
    fn from_auto_detects_binary() {
        let mlp = SimpleMlp::from_dims(&[8, 4], Activation::Tanh);
        let bytes = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_auto(&bytes).unwrap();
        assert_eq!(restored.layers[0].weight, mlp.layers[0].weight);
    }

    #[test]
    fn from_auto_detects_json() {
        let mlp = SimpleMlp::from_dims(&[8, 4], Activation::Tanh);
        let json = mlp.to_json().unwrap();
        let restored = SimpleMlp::from_auto(json.as_bytes()).unwrap();
        assert_eq!(restored.layers[0].weight, mlp.layers[0].weight);
    }

    #[test]
    fn serde_alias_weights_biases() {
        let json = r#"{"layers":[{"weights":[[1.0,2.0],[3.0,4.0]],"biases":[0.1,0.2],"activation":"sigmoid"}]}"#;
        let mlp: SimpleMlp = serde_json::from_str(json).unwrap();
        assert_eq!(mlp.layers[0].weight, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(mlp.layers[0].bias, vec![0.1, 0.2]);
    }

    #[test]
    fn serde_canonical_weight_bias() {
        let json = r#"{"layers":[{"weight":[[1.0,2.0],[3.0,4.0]],"bias":[0.1,0.2],"activation":"relu"}]}"#;
        let mlp: SimpleMlp = serde_json::from_str(json).unwrap();
        assert_eq!(mlp.layers[0].weight, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(mlp.layers[0].activation, Activation::Relu);
    }

    #[test]
    fn binary_trained_model_inference_matches() {
        let mut mlp = SimpleMlp::from_dims(&[4, 2], Activation::Sigmoid);
        let inputs = vec![vec![1.0, 0.0, 0.5, 0.3]];
        let targets = vec![vec![0.8, 0.2]];
        let config = TrainConfig {
            learning_rate: 0.1,
            epochs: 50,
        };
        mlp.train(&inputs, &targets, &config).unwrap();

        let original_output = mlp.forward(&inputs[0]);
        let bytes = mlp.to_binary().unwrap();
        let restored = SimpleMlp::from_binary(&bytes).unwrap();
        let restored_output = restored.forward(&inputs[0]);
        assert_eq!(original_output, restored_output);
    }
}
