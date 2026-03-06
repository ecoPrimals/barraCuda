// SPDX-License-Identifier: AGPL-3.0-or-later
//! LSTM reservoir for recurrent computation.
//!
//! CPU reference implementation with JSON weight serialization.
//! GPU dispatch can come later via `ComputeDispatch`.
//!
//! Provenance: neuralSpring V24 handoff → barracuda nn module.

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Configuration for LSTM reservoir.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmReservoirConfig {
    /// Input feature dimension.
    pub input_size: usize,
    /// Hidden state dimension.
    pub hidden_size: usize,
    /// Number of stacked LSTM layers.
    pub num_layers: usize,
    /// Dropout probability (0.0–1.0) applied between layers (multi-layer only).
    #[serde(default)]
    pub dropout: f64,
}

impl Default for LstmReservoirConfig {
    fn default() -> Self {
        Self {
            input_size: 1,
            hidden_size: 64,
            num_layers: 1,
            dropout: 0.0,
        }
    }
}

/// Hidden and cell state for a single LSTM layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmState {
    /// Hidden state vector (h).
    pub hidden: Vec<f64>,
    /// Cell state vector (c).
    pub cell: Vec<f64>,
}

impl LstmState {
    /// Create zero state for given hidden size.
    #[must_use]
    pub fn zeros(hidden_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_size],
            cell: vec![0.0; hidden_size],
        }
    }
}

/// Per-layer LSTM weights (input gate, forget gate, output gate, cell candidate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmLayerWeights {
    /// `W_ii`, `W_if`, `W_ig`, `W_io`: input projections (4 * `hidden_size` × `input_size`)
    pub w_ih: Vec<Vec<f64>>,
    /// `W_hi`, `W_hf`, `W_hg`, `W_ho`: recurrent projections (4 * `hidden_size` × `hidden_size`)
    pub w_hh: Vec<Vec<f64>>,
    /// `b_i`, `b_f`, `b_g`, `b_o`: biases (4 * `hidden_size`)
    pub bias: Vec<f64>,
}

/// LSTM reservoir with JSON weight serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmReservoir {
    /// Reservoir configuration (sizes, layers, etc.)
    pub config: LstmReservoirConfig,
    /// Per-layer weights (input, forget, cell, output gates)
    pub layers: Vec<LstmLayerWeights>,
}

impl LstmReservoir {
    /// Create new LSTM with random weights (Xavier-like init).
    #[must_use]
    pub fn new(config: LstmReservoirConfig) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut layers = Vec::with_capacity(config.num_layers);

        let input_size = config.input_size;
        let hidden_size = config.hidden_size;

        for layer_idx in 0..config.num_layers {
            let in_dim = if layer_idx == 0 {
                input_size
            } else {
                hidden_size
            };

            let scale_ih = 1.0 / (in_dim as f64).sqrt();
            let scale_hh = 1.0 / (hidden_size as f64).sqrt();

            let mut w_ih = Vec::with_capacity(4 * hidden_size);
            for _ in 0..(4 * hidden_size) {
                let row: Vec<f64> = (0..in_dim)
                    .map(|_| rng.random_range(-1.0..1.0) * scale_ih)
                    .collect();
                w_ih.push(row);
            }

            let mut w_hh = Vec::with_capacity(4 * hidden_size);
            for _ in 0..(4 * hidden_size) {
                let row: Vec<f64> = (0..hidden_size)
                    .map(|_| rng.random_range(-1.0..1.0) * scale_hh)
                    .collect();
                w_hh.push(row);
            }

            let bias: Vec<f64> = (0..(4 * hidden_size)).map(|_| 0.0).collect();

            layers.push(LstmLayerWeights { w_ih, w_hh, bias });
        }

        Self { config, layers }
    }

    /// Construct from JSON string.
    /// # Errors
    /// Returns [`Err`] if JSON parsing fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON string.
    /// # Errors
    /// Returns [`Err`] if JSON serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Single step forward.
    /// Returns output hidden state for the last layer.
    /// # Panics
    /// Panics if `state.len() != self.config.num_layers` or `input.len() != self.config.input_size`.
    #[must_use]
    pub fn forward(&self, input: &[f64], state: &mut [LstmState]) -> Vec<f64> {
        assert_eq!(state.len(), self.config.num_layers);
        assert_eq!(input.len(), self.config.input_size);

        let hidden_size = self.config.hidden_size;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let in_vec = if layer_idx == 0 {
                input.to_vec()
            } else {
                state[layer_idx - 1].hidden.clone()
            };

            let (i, f, g, o) = lstm_gates(layer, &in_vec, &state[layer_idx].hidden);

            for j in 0..hidden_size {
                state[layer_idx].cell[j] = f[j] * state[layer_idx].cell[j] + i[j] * g[j];
                state[layer_idx].hidden[j] = o[j] * state[layer_idx].cell[j].tanh();
            }

            // Dropout between layers (training-time; inference we typically use 0)
            if layer_idx < self.config.num_layers - 1
                && self.config.dropout > 0.0
                && self.config.dropout < 1.0
            {
                for h in &mut state[layer_idx].hidden {
                    *h *= 1.0 - self.config.dropout;
                }
            }
        }

        state[self.config.num_layers - 1].hidden.clone()
    }

    /// Full sequence forward.
    /// Returns (outputs per timestep, final state).
    #[must_use]
    pub fn forward_sequence(
        &self,
        inputs: &[Vec<f64>],
        initial_state: Option<Vec<LstmState>>,
    ) -> (Vec<Vec<f64>>, Vec<LstmState>) {
        let mut state: Vec<LstmState> = initial_state.unwrap_or_else(|| {
            (0..self.config.num_layers)
                .map(|_| LstmState::zeros(self.config.hidden_size))
                .collect()
        });

        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let out = self.forward(input, &mut state);
            outputs.push(out);
        }

        (outputs, state)
    }
}

fn lstm_gates(
    layer: &LstmLayerWeights,
    input: &[f64],
    hidden: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = hidden.len();
    let mut i = vec![0.0; n];
    let mut f = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut o = vec![0.0; n];

    for row_idx in 0..n {
        for (block, out) in [
            (0, &mut i[row_idx]),
            (1, &mut f[row_idx]),
            (2, &mut g[row_idx]),
            (3, &mut o[row_idx]),
        ] {
            let base = block * n + row_idx;
            let mut sum = layer.bias[base];
            for (w, x) in layer.w_ih[base].iter().zip(input) {
                sum += w * x;
            }
            for (w, h) in layer.w_hh[base].iter().zip(hidden) {
                sum += w * h;
            }
            *out = sum;
        }
    }

    sigmoid(&mut i);
    sigmoid(&mut f);
    for v in &mut g {
        *v = v.tanh();
    }
    sigmoid(&mut o);

    (i, f, g, o)
}

fn sigmoid(x: &mut [f64]) {
    for v in x {
        *v = if *v >= 0.0 {
            1.0 / (1.0 + (-*v).exp())
        } else {
            let ez = v.exp();
            ez / (1.0 + ez)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lstm() -> LstmReservoir {
        let config = LstmReservoirConfig {
            input_size: 2,
            hidden_size: 4,
            num_layers: 1,
            dropout: 0.0,
        };
        LstmReservoir::new(config)
    }

    #[test]
    fn test_lstm_forward_single_step() {
        let lstm = make_test_lstm();
        let mut state = vec![LstmState::zeros(4)];
        let input = vec![1.0, 0.5];
        let out = lstm.forward(&input, &mut state);
        assert_eq!(out.len(), 4);
        assert_eq!(state[0].hidden.len(), 4);
        assert_eq!(state[0].cell.len(), 4);
    }

    #[test]
    fn test_lstm_sequence() {
        let lstm = make_test_lstm();
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let (outputs, final_state) = lstm.forward_sequence(&inputs, None);
        assert_eq!(outputs.len(), 3);
        for out in &outputs {
            assert_eq!(out.len(), 4);
        }
        assert_eq!(final_state.len(), 1);
        assert_eq!(final_state[0].hidden.len(), 4);
    }

    #[test]
    fn test_lstm_serde() {
        let lstm = make_test_lstm();
        let json = lstm.to_json().unwrap();
        let lstm2 = LstmReservoir::from_json(&json).unwrap();
        assert_eq!(lstm.config.input_size, lstm2.config.input_size);
        assert_eq!(lstm.config.hidden_size, lstm2.config.hidden_size);
        assert_eq!(lstm.layers.len(), lstm2.layers.len());

        let input = vec![0.5, -0.3];
        let mut s1 = vec![LstmState::zeros(4)];
        let mut s2 = vec![LstmState::zeros(4)];
        let o1 = lstm.forward(&input, &mut s1);
        let o2 = lstm2.forward(&input, &mut s2);
        for (a, b) in o1.iter().zip(&o2) {
            assert!((a - b).abs() < 1e-14);
        }
    }
}
