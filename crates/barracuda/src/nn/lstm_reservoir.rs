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
        let mut out = vec![0.0; self.config.hidden_size];
        self.forward_into(input, state, &mut out);
        out
    }

    /// Single step forward, writing the output into a caller-provided buffer.
    ///
    /// Avoids the per-timestep allocation of [`forward`]. `output` must have
    /// length `>= hidden_size`; only the first `hidden_size` elements are written.
    ///
    /// # Panics
    /// Panics if `state.len() != self.config.num_layers`, `input.len() != self.config.input_size`,
    /// or `output.len() < self.config.hidden_size`.
    pub fn forward_into(&self, input: &[f64], state: &mut [LstmState], output: &mut [f64]) {
        let mut gates = GateBuffers::new(self.config.hidden_size);
        self.forward_core(input, state, output, &mut gates);
    }

    /// Full sequence forward.
    /// Returns (outputs per timestep, final state).
    ///
    /// Uses pre-allocated gate and output buffers internally to avoid
    /// per-timestep allocation.
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

        let hs = self.config.hidden_size;
        let mut buf = vec![0.0; hs];
        let mut gates = GateBuffers::new(hs);
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            self.forward_core(input, &mut state, &mut buf, &mut gates);
            outputs.push(buf[..hs].to_vec());
        }

        (outputs, state)
    }

    fn forward_core(
        &self,
        input: &[f64],
        state: &mut [LstmState],
        output: &mut [f64],
        gates: &mut GateBuffers,
    ) {
        assert_eq!(state.len(), self.config.num_layers);
        assert_eq!(input.len(), self.config.input_size);
        assert!(
            output.len() >= self.config.hidden_size,
            "output buffer too small: {} < {}",
            output.len(),
            self.config.hidden_size
        );

        let hidden_size = self.config.hidden_size;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer_idx == 0 {
                lstm_gates_into(layer, input, &state[layer_idx].hidden, gates);
            } else {
                let (prev, curr) = state.split_at_mut(layer_idx);
                lstm_gates_into(layer, &prev[layer_idx - 1].hidden, &curr[0].hidden, gates);
            }

            for j in 0..hidden_size {
                state[layer_idx].cell[j] =
                    gates.f[j].mul_add(state[layer_idx].cell[j], gates.i[j] * gates.g[j]);
                state[layer_idx].hidden[j] = gates.o[j] * state[layer_idx].cell[j].tanh();
            }

            if layer_idx < self.config.num_layers - 1
                && self.config.dropout > 0.0
                && self.config.dropout < 1.0
            {
                for h in &mut state[layer_idx].hidden {
                    *h *= 1.0 - self.config.dropout;
                }
            }
        }

        output[..hidden_size].copy_from_slice(&state[self.config.num_layers - 1].hidden);
    }
}

/// Pre-allocated scratch buffers for LSTM gate computation.
/// Avoids 4 × `Vec<f64>` allocations per layer per timestep.
struct GateBuffers {
    i: Vec<f64>,
    f: Vec<f64>,
    g: Vec<f64>,
    o: Vec<f64>,
}

impl GateBuffers {
    fn new(hidden_size: usize) -> Self {
        Self {
            i: vec![0.0; hidden_size],
            f: vec![0.0; hidden_size],
            g: vec![0.0; hidden_size],
            o: vec![0.0; hidden_size],
        }
    }

    fn reset(&mut self) {
        self.i.fill(0.0);
        self.f.fill(0.0);
        self.g.fill(0.0);
        self.o.fill(0.0);
    }
}

fn lstm_gates_into(layer: &LstmLayerWeights, input: &[f64], hidden: &[f64], buf: &mut GateBuffers) {
    buf.reset();
    let n = hidden.len();

    for row_idx in 0..n {
        for (block, out) in [
            (0, &mut buf.i[row_idx]),
            (1, &mut buf.f[row_idx]),
            (2, &mut buf.g[row_idx]),
            (3, &mut buf.o[row_idx]),
        ] {
            let base = block * n + row_idx;
            let mut sum = layer.bias[base];
            for (w, x) in layer.w_ih[base].iter().zip(input) {
                sum = w.mul_add(*x, sum);
            }
            for (w, h) in layer.w_hh[base].iter().zip(hidden) {
                sum = w.mul_add(*h, sum);
            }
            *out = sum;
        }
    }

    sigmoid(&mut buf.i);
    sigmoid(&mut buf.f);
    for v in &mut buf.g {
        *v = v.tanh();
    }
    sigmoid(&mut buf.o);
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
    fn test_lstm_forward_into_matches_forward() {
        let lstm = make_test_lstm();
        let input = vec![1.0, 0.5];

        let mut s1 = vec![LstmState::zeros(4)];
        let out_alloc = lstm.forward(&input, &mut s1);

        let mut s2 = vec![LstmState::zeros(4)];
        let mut buf = vec![0.0; 4];
        lstm.forward_into(&input, &mut s2, &mut buf);

        for (a, b) in out_alloc.iter().zip(&buf) {
            assert!((a - b).abs() < 1e-15, "forward vs forward_into mismatch");
        }
        for (a, b) in s1[0].hidden.iter().zip(&s2[0].hidden) {
            assert!((a - b).abs() < 1e-15);
        }
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
