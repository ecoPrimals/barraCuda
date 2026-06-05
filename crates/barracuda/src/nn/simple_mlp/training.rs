// SPDX-License-Identifier: AGPL-3.0-or-later
//! Training logic for `SimpleMlp` — mini-batch SGD with backpropagation.

use super::{SimpleMlp, activation_derivative, apply_activation};

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

impl SimpleMlp {
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
                let (activations, pre_activations) = self.forward_with_cache(input);

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
                    let grad: Vec<f64> =
                        delta.iter().zip(&act_deriv).map(|(d, a)| d * a).collect();

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
    pub(super) fn forward_with_cache(
        &self,
        input: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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
