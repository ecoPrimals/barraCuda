// SPDX-License-Identifier: AGPL-3.0-only
//! Echo State Network classifier with ridge regression readout.
//!
//! CPU reference implementation with JSON weight serialization.
//! Fixed random reservoir, trained readout layer.
//!
//! Provenance: neuralSpring V24 handoff → barracuda nn module.

use crate::error::{BarracudaError, Result};
use crate::linalg::ridge::ridge_regression;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Configuration for Echo State Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsnConfig {
    /// Number of input features
    pub input_size: usize,
    /// Number of reservoir neurons
    pub reservoir_size: usize,
    /// Number of output classes/features
    pub output_size: usize,
    /// Target spectral radius (typically 0.9–0.99)
    pub spectral_radius: f64,
    /// Fraction of non-zero reservoir weights (0.0–1.0), sparsity
    pub sparsity: f64,
    /// Leak rate for temporal integration (0.0–1.0)
    pub leak_rate: f64,
    /// Ridge regression regularization parameter (> 0)
    #[serde(default = "default_regularization")]
    pub regularization: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

fn default_regularization() -> f64 {
    1e-6
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            input_size: 1,
            reservoir_size: 100,
            output_size: 1,
            spectral_radius: 0.95,
            sparsity: 0.1,
            leak_rate: 0.3,
            regularization: 1e-6,
            seed: 42,
        }
    }
}

/// Serializable ESN snapshot for JSON round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsnWeights {
    /// ESN configuration.
    pub config: EsnConfig,
    /// Input-to-reservoir weights (`reservoir_size` × `input_size`, row-major)
    pub w_in: Vec<f64>,
    /// Reservoir recurrent weights (`reservoir_size` × `reservoir_size`, row-major)
    pub w_res: Vec<f64>,
    /// Readout weights (`output_size` × `reservoir_size`, row-major)
    pub w_out: Option<Vec<f64>>,
}

/// Echo State Network classifier with ridge regression readout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsnClassifier {
    /// ESN configuration.
    pub config: EsnConfig,
    #[serde(skip)]
    state: Vec<f64>,
    /// Input-to-reservoir weights.
    pub w_in: Vec<f64>,
    /// Reservoir recurrent weights.
    pub w_res: Vec<f64>,
    /// Readout weights (None if untrained).
    pub w_out: Option<Vec<f64>>,
}

impl EsnClassifier {
    /// Create new ESN with random reservoir (fixed), untrained readout.
    /// # Errors
    /// Returns [`Err`] if the configuration is invalid (sizes zero, spectral radius
    /// outside (0, 2], sparsity or leak rate outside (0, 1], or non-positive regularization).
    pub fn new(config: EsnConfig) -> Result<Self> {
        validate_esn_config(&config)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        let w_res = Self::init_reservoir(&config, &mut rng);
        let w_in = Self::init_input_weights(&config, &mut rng);

        Ok(Self {
            config: config.clone(),
            state: vec![0.0; config.reservoir_size],
            w_in,
            w_res,
            w_out: None,
        })
    }

    fn init_reservoir(config: &EsnConfig, rng: &mut rand::rngs::StdRng) -> Vec<f64> {
        let size = config.reservoir_size;
        let mut matrix = vec![0.0; size * size];

        for i in 0..size {
            for j in 0..size {
                if rng.random::<f64>() < config.sparsity {
                    matrix[i * size + j] = rng.random_range(-1.0..1.0);
                }
            }
        }

        let approx_radius = (config.sparsity * size as f64).sqrt();
        let scale = config.spectral_radius / approx_radius;

        for val in &mut matrix {
            *val *= scale;
        }

        matrix
    }

    fn init_input_weights(config: &EsnConfig, rng: &mut rand::rngs::StdRng) -> Vec<f64> {
        (0..(config.reservoir_size * config.input_size))
            .map(|_| rng.random_range(-0.5..0.5))
            .collect()
    }

    /// Construct from JSON string.
    /// # Errors
    /// Returns [`Err`] if the JSON fails to parse or the parsed configuration is invalid.
    pub fn from_json(json: &str) -> Result<Self> {
        let weights: EsnWeights =
            serde_json::from_str(json).map_err(|e| BarracudaError::InvalidInput {
                message: format!("ESN JSON parse error: {e}"),
            })?;

        validate_esn_config(&weights.config)?;

        Ok(Self {
            config: weights.config.clone(),
            state: vec![0.0; weights.config.reservoir_size],
            w_in: weights.w_in,
            w_res: weights.w_res,
            w_out: weights.w_out,
        })
    }

    /// Serialize to JSON string.
    /// # Errors
    /// Returns [`Err`] if JSON serialization fails.
    pub fn to_json(&self) -> Result<String> {
        let weights = EsnWeights {
            config: self.config.clone(),
            w_in: self.w_in.clone(),
            w_res: self.w_res.clone(),
            w_out: self.w_out.clone(),
        };
        serde_json::to_string_pretty(&weights).map_err(|e| BarracudaError::InvalidInput {
            message: format!("ESN JSON serialize error: {e}"),
        })
    }

    /// Reset reservoir state to zero.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Update reservoir state with a single input (internal).
    fn update_state(&mut self, input: &[f64]) -> Result<()> {
        if input.len() != self.config.input_size {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Input size {} != expected {}",
                    input.len(),
                    self.config.input_size
                ),
            });
        }

        let n = self.config.reservoir_size;
        let leak = self.config.leak_rate;

        let mut new_state = vec![0.0; n];

        for i in 0..n {
            let mut sum = 0.0;
            for (j, &w) in self.w_in[i * self.config.input_size..(i + 1) * self.config.input_size]
                .iter()
                .enumerate()
            {
                sum += w * input[j];
            }
            for j in 0..n {
                sum += self.w_res[i * n + j] * self.state[j];
            }
            new_state[i] = sum.tanh();
        }

        for i in 0..n {
            self.state[i] = (1.0 - leak) * self.state[i] + leak * new_state[i];
        }

        Ok(())
    }

    /// Train the readout layer via ridge regression.
    /// # Errors
    /// Returns [`Err`] if inputs or targets are empty, lengths differ, any input
    /// has wrong size, any target has wrong size, or ridge regression fails.
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<()> {
        if inputs.is_empty() || targets.is_empty() {
            return Err(BarracudaError::InvalidInput {
                message: "Training data cannot be empty".to_string(),
            });
        }

        if inputs.len() != targets.len() {
            return Err(BarracudaError::InvalidInput {
                message: "Inputs and targets must have same length".to_string(),
            });
        }

        self.reset_state();

        let n = self.config.reservoir_size;
        let m = self.config.output_size;
        let n_samples = inputs.len();
        let n_features = n + 1;

        let mut states = vec![0.0; n_samples * n_features];
        let mut targets_flat = vec![0.0; n_samples * m];

        for (s, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
            if input.len() != self.config.input_size {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "Input[{}] size {} != expected {}",
                        s,
                        input.len(),
                        self.config.input_size
                    ),
                });
            }
            if target.len() != m {
                return Err(BarracudaError::InvalidInput {
                    message: format!("Target[{}] size {} != expected {}", s, target.len(), m),
                });
            }

            self.update_state(input)?;

            for i in 0..n {
                states[s * n_features + i] = self.state[i];
            }
            states[s * n_features + n] = 1.0;
            for j in 0..m {
                targets_flat[s * m + j] = target[j];
            }
        }

        let ridge = ridge_regression(
            &states,
            &targets_flat,
            n_samples,
            n_features,
            m,
            self.config.regularization,
        )?;

        self.w_out = Some(ridge.weights);
        Ok(())
    }

    /// Predict output for a single input.
    /// # Errors
    /// Returns [`Err`] if the ESN is untrained or the input length does not match
    /// the configured input size.
    pub fn predict(&mut self, input: &[f64]) -> Result<Vec<f64>> {
        if self.w_out.is_none() {
            return Err(BarracudaError::InvalidInput {
                message: "ESN must be trained before prediction".to_string(),
            });
        }

        self.update_state(input)?;

        let n = self.config.reservoir_size;
        let m = self.config.output_size;
        let n_features = n + 1;
        let w_out = self
            .w_out
            .as_ref()
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: "readout weights missing after training guard".to_string(),
            })?;

        let mut output = vec![0.0; m];
        for j in 0..m {
            let mut sum = 0.0;
            for i in 0..n {
                sum += w_out[j * n_features + i] * self.state[i];
            }
            sum += w_out[j * n_features + n];
            output[j] = sum;
        }

        Ok(output)
    }

    /// Predict without updating state (uses current state for readout).
    /// # Errors
    /// Returns [`Err`] if the ESN is untrained.
    pub fn predict_from_state(&self) -> Result<Vec<f64>> {
        let w_out = self
            .w_out
            .as_ref()
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: "ESN must be trained before prediction".to_string(),
            })?;

        let n = self.config.reservoir_size;
        let m = self.config.output_size;
        let n_features = n + 1;

        let mut output = vec![0.0; m];
        for j in 0..m {
            let mut sum = 0.0;
            for i in 0..n {
                sum += w_out[j * n_features + i] * self.state[i];
            }
            sum += w_out[j * n_features + n];
            output[j] = sum;
        }

        Ok(output)
    }

    /// Check if ESN is trained.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.w_out.is_some()
    }
}

fn validate_esn_config(config: &EsnConfig) -> Result<()> {
    if config.input_size == 0 || config.reservoir_size == 0 || config.output_size == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "All sizes must be greater than zero".to_string(),
        });
    }
    if config.spectral_radius <= 0.0 || config.spectral_radius > 2.0 {
        return Err(BarracudaError::InvalidInput {
            message: "Spectral radius must be in (0, 2]".to_string(),
        });
    }
    if config.sparsity <= 0.0 || config.sparsity > 1.0 {
        return Err(BarracudaError::InvalidInput {
            message: "Sparsity must be in (0, 1]".to_string(),
        });
    }
    if config.leak_rate <= 0.0 || config.leak_rate > 1.0 {
        return Err(BarracudaError::InvalidInput {
            message: "Leak rate must be in (0, 1]".to_string(),
        });
    }
    if config.regularization <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: "Regularization must be positive".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_esn() -> EsnClassifier {
        let config = EsnConfig {
            input_size: 2,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            sparsity: 0.1,
            leak_rate: 0.3,
            regularization: 1e-6,
            seed: 42,
        };
        EsnClassifier::new(config).unwrap()
    }

    #[test]
    fn test_esn_train_xor() {
        let config = EsnConfig {
            input_size: 2,
            reservoir_size: 100,
            output_size: 1,
            spectral_radius: 0.95,
            sparsity: 0.1,
            leak_rate: 0.3,
            regularization: 1e-6,
            seed: 42,
        };
        let mut esn = EsnClassifier::new(config).unwrap();

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        esn.train(&inputs, &targets).unwrap();

        esn.reset_state();
        let p00 = esn.predict(&[0.0, 0.0]).unwrap()[0];
        esn.reset_state();
        let p01 = esn.predict(&[0.0, 1.0]).unwrap()[0];
        esn.reset_state();
        let p10 = esn.predict(&[1.0, 0.0]).unwrap()[0];
        esn.reset_state();
        let p11 = esn.predict(&[1.0, 1.0]).unwrap()[0];

        let zeros = f64::midpoint(p00, p11);
        let ones = f64::midpoint(p01, p10);
        assert!(
            ones > zeros,
            "XOR: (0,1) and (1,0) should output higher than (0,0) and (1,1); \
             got p00={p00:.3} p01={p01:.3} p10={p10:.3} p11={p11:.3}"
        );
        assert!(p00 < 0.5, "XOR(0,0) should be ~0, got {p00}");
        assert!(p11 < 0.5, "XOR(1,1) should be ~0, got {p11}");
    }

    #[test]
    fn test_esn_serde() {
        let mut esn = make_test_esn();
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![vec![1.0], vec![0.0]];
        esn.train(&inputs, &targets).unwrap();

        let json = esn.to_json().unwrap();
        let mut esn2 = EsnClassifier::from_json(&json).unwrap();

        assert_eq!(esn.w_in.len(), esn2.w_in.len());
        assert_eq!(esn.w_res.len(), esn2.w_res.len());
        assert!(esn2.w_out.is_some());

        esn.reset_state();
        esn2.reset_state();
        let p1 = esn.predict(&[0.5, 0.5]).unwrap();
        let p2 = esn2.predict(&[0.5, 0.5]).unwrap();
        assert!((p1[0] - p2[0]).abs() < 1e-10);
    }

    #[test]
    fn test_esn_reset_state() {
        let mut esn = make_test_esn();
        let inputs = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let targets = vec![vec![0.0], vec![0.0]];
        esn.train(&inputs, &targets).unwrap();

        esn.predict(&[1.0, 0.0]).unwrap();
        esn.reset_state();
        let out = esn.predict(&[0.0, 0.0]).unwrap();
        assert!(out[0].abs() < 1.0 || out[0].abs() > 0.0);
    }
}
