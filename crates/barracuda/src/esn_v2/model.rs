// SPDX-License-Identifier: AGPL-3.0-only
//! ESN model - training, prediction, and weight management

use crate::device::{Auto, Device, WgpuDevice, WorkloadHint};
use crate::error::{BarracudaError, Result as BarracudaResult};
use crate::linalg::solve_f64_cpu;
use crate::tensor::Tensor;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

use super::config::{ESNConfig, expect_size, validate_config};
use super::npu::{NpuReadoutWeights, quantize_affine_i8_f64};

/// Serializable ESN weight snapshot for cross-run / cross-device deployment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportedWeights {
    /// Input-to-reservoir weights (`reservoir_size` × `input_size`).
    pub w_in: Vec<f32>,
    /// Reservoir recurrent weights (`reservoir_size` × `reservoir_size`).
    pub w_res: Vec<f32>,
    /// Readout weights (`reservoir_size` × `output_size`), None if untrained.
    pub w_out: Option<Vec<f32>>,
    /// Input dimension for cross-device reconstruction.
    #[serde(default)]
    pub input_size: usize,
    /// Reservoir size for cross-device reconstruction.
    #[serde(default)]
    pub reservoir_size: usize,
    /// Output dimension for cross-device reconstruction.
    #[serde(default)]
    pub output_size: usize,
    /// Leak rate for cross-device reconstruction.
    #[serde(default)]
    pub leak_rate: f32,
    /// Optional head labels for multi-head ESN.
    #[serde(default)]
    pub head_labels: Vec<String>,
}

impl ExportedWeights {
    /// Migrate a single-head weight snapshot to a multi-head ESN.
    /// The reservoir weights (`w_in`, `w_res`) are preserved unchanged.
    /// If `w_out` was trained for 1 head, it is replicated across all
    /// `new_output_size` heads. If `w_out` is `None`, it remains `None`.
    /// hotSpring v0.6.15 absorption: enables 1-head → 11-head migration.
    /// # Errors
    /// Returns [`Err`] if `w_out` length is not divisible by `reservoir_size`, or
    /// if `w_out` length divided by `reservoir_size` is zero.
    pub fn migrate_to_multi_head(
        &self,
        reservoir_size: usize,
        new_output_size: usize,
    ) -> BarracudaResult<Self> {
        let mut migrated = self.clone();
        migrated.output_size = new_output_size;
        if let Some(ref w_out) = self.w_out {
            let old_outputs = w_out.len() / reservoir_size;
            if old_outputs == 0 || w_out.len() % reservoir_size != 0 {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "w_out length {} not divisible by reservoir_size {}",
                        w_out.len(),
                        reservoir_size
                    ),
                });
            }
            if old_outputs == new_output_size {
                return Ok(migrated);
            }
            let mut new_w_out = Vec::with_capacity(new_output_size * reservoir_size);
            for head in 0..new_output_size {
                let src_head = head % old_outputs;
                let src_start = src_head * reservoir_size;
                new_w_out.extend_from_slice(&w_out[src_start..src_start + reservoir_size]);
            }
            migrated.w_out = Some(new_w_out);
        }
        Ok(migrated)
    }
}

/// Hardware-Agnostic Echo State Network
///
/// **Uses `BarraCuda` Tensors** - Works on CPU, GPU, NPU!
pub struct ESN {
    pub(super) config: ESNConfig,

    pub(super) w_in: Tensor,
    pub(super) w_res: Tensor,
    pub(super) w_out: Option<Tensor>,
    pub(super) state: Tensor,
    pub(super) device: Arc<WgpuDevice>,
    pub(super) trained: bool,
}

impl ESN {
    /// Create a new Echo State Network
    /// **Hardware-agnostic** - Auto-detects best device!
    /// # Errors
    /// Returns [`Err`] if config validation fails (invalid sizes, spectral radius,
    /// connectivity, leak rate, or regularization), if device creation fails, if
    /// reservoir or input weight initialization fails, or if buffer allocation fails.
    pub async fn new(config: ESNConfig) -> BarracudaResult<Self> {
        validate_config(&config)?;

        let device = Auto::new().await?;

        let w_res = Self::init_reservoir(&config, &device).await?;
        let w_in = Self::init_input_weights(&config, &device).await?;
        let state = Tensor::zeros_on(vec![config.reservoir_size, 1], device.clone()).await?;

        Ok(Self {
            config,
            w_in,
            w_res,
            w_out: None,
            state,
            device,
            trained: false,
        })
    }

    /// Initialize reservoir weights
    async fn init_reservoir(
        config: &ESNConfig,
        device: &Arc<WgpuDevice>,
    ) -> BarracudaResult<Tensor> {
        let size = config.reservoir_size;

        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let mut matrix = vec![0.0; size * size];

        for i in 0..size {
            for j in 0..size {
                if rng.random::<f32>() < config.connectivity {
                    matrix[i * size + j] = rng.random_range(-1.0..1.0);
                }
            }
        }

        let approx_radius = (config.connectivity * size as f32).sqrt();
        let scale = config.spectral_radius / approx_radius;

        for val in &mut matrix {
            *val *= scale;
        }

        Tensor::from_vec_on(matrix, vec![size, size], device.clone()).await
    }

    /// Initialize input weights
    async fn init_input_weights(
        config: &ESNConfig,
        device: &Arc<WgpuDevice>,
    ) -> BarracudaResult<Tensor> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed + 1);
        let weights: Vec<f32> = (0..(config.reservoir_size * config.input_size))
            .map(|_| rng.random::<f32>() - 0.5)
            .collect();

        Tensor::from_vec_on(
            weights,
            vec![config.reservoir_size, config.input_size],
            device.clone(),
        )
        .await
    }

    /// Record a device preference for future migration.
    ///
    /// The preference is stored and reported by [`query_device`](Self::query_device),
    /// but live tensor migration between devices is not yet implemented —
    /// the ESN continues using the device it was created on.
    #[must_use]
    pub fn prefer_device(self, device: Device) -> Self {
        tracing::debug!(preferred = %device, current = %self.query_device(), "device preference recorded");
        self
    }

    /// Record a workload hint for smart device routing.
    ///
    /// Logs the recommended device based on the hint. Like `prefer_device`,
    /// the hint is advisory — live migration is not yet implemented.
    #[must_use]
    pub fn with_hint(self, hint: WorkloadHint) -> Self {
        let preferred = Device::select_for_workload(&hint);
        tracing::debug!(?hint, %preferred, "workload hint recorded");
        self
    }

    /// Reset reservoir state to zero
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails or the device is lost.
    pub async fn reset_state(&mut self) -> BarracudaResult<()> {
        self.state =
            Tensor::zeros_on(vec![self.config.reservoir_size, 1], self.device.clone()).await?;
        Ok(())
    }

    /// Update reservoir state with a single input.
    /// Accepts column vectors of shape `[input_size, 1]`. Row vectors
    /// `[1, input_size]` and flat vectors `[input_size]` are automatically
    /// reshaped to column form for convenience.
    /// # Errors
    /// Returns [`Err`] if input shape does not match `[input_size, 1]`, `[1, input_size]`,
    /// or `[input_size]`, or if any tensor operation (reshape, matmul, add, tanh, etc.)
    /// fails due to shape mismatch or device error.
    pub async fn update(&mut self, input: &Tensor) -> BarracudaResult<Tensor> {
        let input = match input.shape() {
            [n, 1] if *n == self.config.input_size => input.clone(),
            [1, n] if *n == self.config.input_size => {
                input.reshape(vec![self.config.input_size, 1])?
            }
            [n] if *n == self.config.input_size => {
                input.reshape(vec![self.config.input_size, 1])?
            }
            other => {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "Input tensor shape mismatch: expected [{}, 1] (or [1, {}] or [{}]), got {other:?}",
                        self.config.input_size, self.config.input_size, self.config.input_size,
                    ),
                });
            }
        };
        let input = &input;

        let leak = self.config.leak_rate;

        let input_contrib = self.w_in.clone().matmul(input)?;
        let recurrent_contrib = self.w_res.clone().matmul(&self.state)?;
        let combined = input_contrib.add(&recurrent_contrib)?;
        let activated = combined.tanh()?;

        let old_state_scaled = self.state.mul_scalar(1.0 - leak)?;
        let activated_scaled = activated.mul_scalar(leak)?;
        let new_state = old_state_scaled.add(&activated_scaled)?;

        self.state = new_state.clone();
        Ok(new_state)
    }

    /// Train the ESN readout layer
    /// # Errors
    /// Returns [`Err`] if inputs or targets are empty, if inputs and targets have
    /// different lengths, if any input or target has wrong size (via `expect_size`),
    /// if buffer allocation or tensor operations fail, or if ridge regression solve fails.
    pub async fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> BarracudaResult<f32> {
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

        let mut all_states = Vec::new();
        let mut all_targets = Vec::new();

        for (input_seq, target_seq) in inputs.iter().zip(targets.iter()) {
            expect_size("Input", self.config.input_size, input_seq.len())?;
            expect_size("Target", self.config.output_size, target_seq.len())?;

            let input_tensor = Tensor::from_vec_on(
                input_seq.clone(),
                vec![self.config.input_size, 1],
                self.device.clone(),
            )
            .await?;

            let state = self.update(&input_tensor).await?;

            all_states.push(state.to_vec()?);
            all_targets.extend_from_slice(target_seq);
        }

        let n_samples = all_states.len();
        let n = self.config.reservoir_size;
        let m = self.config.output_size;

        let states_flat: Vec<f32> = all_states.into_iter().flatten().collect();

        let states_tensor =
            Tensor::from_vec_on(states_flat, vec![n_samples, n], self.device.clone()).await?;

        let targets_tensor =
            Tensor::from_vec_on(all_targets, vec![n_samples, m], self.device.clone()).await?;

        let w_out = self
            .ridge_regression_solve(&states_tensor, &targets_tensor)
            .await?;

        let predictions = states_tensor.clone().matmul(&w_out)?;
        let diff = predictions.sub(&targets_tensor)?;
        let error_vec = diff.to_vec()?;
        let error: f32 = error_vec.iter().map(|x| x * x).sum::<f32>() / n_samples as f32;

        self.w_out = Some(w_out);
        self.trained = true;

        Ok(error.sqrt())
    }

    /// Train the readout layer using matrix ridge regression (closed-form).
    /// Implements `W_out` = Y * X^T * (X * X^T + lambda * I)^{-1} using CPU solve.
    /// Uses `solve_f64_cpu` from linalg for the matrix solve (ESN matrices are small).
    /// # Arguments
    /// * `states` - State matrix (`reservoir_size` × `n_samples`), row-major
    /// * `targets` - Target matrix (`output_size` × `n_samples`), row-major
    /// * `lambda` - Ridge regularization parameter (> 0)
    /// # Errors
    /// Returns [`Err`] if states are empty, states length is not divisible by
    /// `reservoir_size`, targets length does not match `output_size * n_samples`,
    /// lambda is not positive, the CPU matrix solve fails, or tensor creation fails.
    pub fn train_ridge_regression(
        &mut self,
        states: &[f64],
        targets: &[f64],
        lambda: f64,
    ) -> BarracudaResult<()> {
        let n = self.config.reservoir_size;
        let m = self.config.output_size;

        if states.is_empty() {
            return Err(BarracudaError::InvalidInput {
                message: "States cannot be empty".to_string(),
            });
        }
        if !states.len().is_multiple_of(n) {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "States length {} must be divisible by reservoir_size {}",
                    states.len(),
                    n
                ),
            });
        }
        let n_samples = states.len() / n;
        expect_size("Targets", m * n_samples, targets.len())?;

        if lambda <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "Lambda must be positive".to_string(),
            });
        }

        let x = states;
        let y = targets;

        let mut m_mat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x[i * n_samples + k] * x[j * n_samples + k];
                }
                m_mat[i * n + j] = sum;
            }
            m_mat[i * n + i] += lambda;
        }

        let mut b_mat = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x[i * n_samples + k] * y[j * n_samples + k];
                }
                b_mat[i * m + j] = sum;
            }
        }

        let mut w_out_t = vec![0.0; n * m];
        for j in 0..m {
            let b_col: Vec<f64> = (0..n).map(|i| b_mat[i * m + j]).collect();
            let w_col = solve_f64_cpu(&m_mat, &b_col, n)?;
            for (i, &w) in w_col.iter().enumerate() {
                w_out_t[i * m + j] = w;
            }
        }

        let w_out_f32: Vec<f32> = w_out_t.iter().map(|&x| x as f32).collect();
        self.w_out = Some(Tensor::from_data(
            &w_out_f32,
            vec![n, m],
            self.device.clone(),
        )?);
        self.trained = true;

        Ok(())
    }

    /// Solve ridge regression using gradient descent.
    ///
    /// Iteration count scales with reservoir size: small reservoirs (< 50)
    /// converge in ~50 steps; larger ones use proportionally more.
    async fn ridge_regression_solve(
        &self,
        states: &Tensor,
        targets: &Tensor,
    ) -> BarracudaResult<Tensor> {
        let n = self.config.reservoir_size;
        let m = self.config.output_size;

        let mut w_out = Tensor::zeros_on(vec![n, m], self.device.clone()).await?;

        let learning_rate = self.config.sgd_learning_rate as f32;
        let iterations = (n * 2).clamp(
            self.config.sgd_min_iterations,
            self.config.sgd_max_iterations,
        );
        let lambda = self.config.regularization;

        for _iter in 0..iterations {
            let predictions = states.clone().matmul(&w_out)?;
            let diff = predictions.sub(targets)?;

            let states_t = states.transpose()?;
            let grad = states_t.clone().matmul(&diff)?;
            let reg_term = w_out.mul_scalar(lambda)?;
            let total_grad = grad.add(&reg_term)?;
            let scaled_grad = total_grad.mul_scalar(learning_rate)?;

            w_out = w_out.sub(&scaled_grad)?;
        }

        Ok(w_out)
    }

    /// Predict on new input sequence
    /// # Errors
    /// Returns [`Err`] with the same conditions as [`predict_return_state`](Self::predict_return_state).
    pub async fn predict(&mut self, input: &[f32]) -> BarracudaResult<Vec<f32>> {
        let (output, _state) = self.predict_return_state(input).await?;
        Ok(output)
    }

    /// Predict and return both output AND raw reservoir state.
    /// # Errors
    /// Returns [`Err`] if the ESN is not trained, if input length does not match
    /// `input_size`, if buffer allocation or tensor operations fail during update,
    /// or if readout weights are missing (untrained ESN).
    pub async fn predict_return_state(
        &mut self,
        input: &[f32],
    ) -> BarracudaResult<(Vec<f32>, Vec<f32>)> {
        if !self.trained {
            return Err(BarracudaError::InvalidInput {
                message: "ESN must be trained before prediction".to_string(),
            });
        }

        expect_size("Input", self.config.input_size, input.len())?;

        let input_tensor = Tensor::from_vec_on(
            input.to_vec(),
            vec![self.config.input_size, 1],
            self.device.clone(),
        )
        .await?;

        let state = self.update(&input_tensor).await?;
        let raw_state = state.to_vec()?;

        let w_out =
            self.w_out
                .as_ref()
                .ok_or_else(|| crate::error::BarracudaError::InvalidOperation {
                    op: "ESN::predict_return_state".to_string(),
                    reason: "ESN has not been trained yet — call train() first".to_string(),
                })?;
        let output = w_out.transpose()?.matmul(&state)?;

        Ok((output.to_vec()?, raw_state))
    }

    /// Replace the readout weights without retraining the reservoir.
    /// Weights must have shape `[reservoir_size, output_size]` — the same
    /// layout produced by `train()` and `train_ridge_regression()`.
    /// Predict transposes internally: `output = W_out^T @ state`.
    /// # Errors
    /// Returns [`Err`] if weight tensor shape is not `[reservoir_size, output_size]`.
    pub fn set_readout_weights(&mut self, weights: Tensor) -> BarracudaResult<()> {
        let expected = [self.config.reservoir_size, self.config.output_size];
        if weights.shape() != expected {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Readout weight shape mismatch: expected {:?}, got {:?}",
                    expected,
                    weights.shape()
                ),
            });
        }
        self.w_out = Some(weights);
        self.trained = true;
        Ok(())
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &ESNConfig {
        &self.config
    }

    /// Check if ESN is trained
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get current reservoir state
    #[must_use]
    pub fn state(&self) -> &Tensor {
        &self.state
    }

    /// Export readout weights as int8-quantized NPU format.
    /// Requires the ESN to be trained. Converts f32 readout weights to f64,
    /// applies affine quantization, and returns `NpuReadoutWeights` suitable
    /// for NPU deployment (e.g. Akida AKD1000 FC layer).
    /// # Errors
    /// Returns [`Err`] if the ESN has not been trained yet, or if reading
    /// readout weights to host fails.
    pub fn to_npu_weights(&self) -> BarracudaResult<NpuReadoutWeights> {
        let w_out =
            self.w_out
                .as_ref()
                .ok_or_else(|| crate::error::BarracudaError::InvalidOperation {
                    op: "ESN::to_npu_weights".to_string(),
                    reason: "ESN has not been trained yet — call train() first".to_string(),
                })?;

        let w_out_f32 = w_out.to_vec()?;
        let w_out_f64: Vec<f64> = w_out_f32.iter().map(|&x| f64::from(x)).collect();

        let (weights_i8, scale, zero_point) = quantize_affine_i8_f64(&w_out_f64);

        Ok(NpuReadoutWeights {
            weights_i8,
            scale,
            zero_point,
            input_dim: self.config.reservoir_size,
            output_dim: self.config.output_size,
        })
    }

    /// Export all ESN weights as flat f32 vectors for cross-device deployment.
    /// # Errors
    /// Returns [`Err`] if reading any weight tensor to host fails (e.g. device lost,
    /// buffer mapping failure).
    pub fn export_weights(&self) -> BarracudaResult<ExportedWeights> {
        let w_in_data = self.w_in.to_vec()?;
        let w_res_data = self.w_res.to_vec()?;
        let w_out_data = self.w_out.as_ref().map(Tensor::to_vec).transpose()?;
        Ok(ExportedWeights {
            w_in: w_in_data,
            w_res: w_res_data,
            w_out: w_out_data,
            input_size: self.config.input_size,
            reservoir_size: self.config.reservoir_size,
            output_size: self.config.output_size,
            leak_rate: self.config.leak_rate,
            head_labels: Vec::new(),
        })
    }

    /// Import pre-trained weights
    /// # Errors
    /// Returns [`Err`] if `w_in`, `w_res`, or `w_out` (when provided) have wrong
    /// lengths for the expected shapes, or if tensor creation from data fails.
    pub fn import_weights(
        &mut self,
        w_in: &[f32],
        w_res: &[f32],
        w_out: Option<&[f32]>,
    ) -> BarracudaResult<()> {
        let rs = self.config.reservoir_size;
        let is = self.config.input_size;
        let os = self.config.output_size;

        expect_size("w_in", rs * is, w_in.len())?;
        expect_size("w_res", rs * rs, w_res.len())?;

        self.w_in = Tensor::from_data(w_in, vec![rs, is], self.device.clone())?;
        self.w_res = Tensor::from_data(w_res, vec![rs, rs], self.device.clone())?;

        if let Some(wo) = w_out {
            expect_size("w_out", rs * os, wo.len())?;
            self.w_out = Some(Tensor::from_data(wo, vec![rs, os], self.device.clone())?);
            self.trained = true;
        }
        Ok(())
    }

    /// Get device query
    #[must_use]
    pub fn query_device(&self) -> Device {
        match self.device.device_type() {
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu => Device::GPU,
            wgpu::DeviceType::VirtualGpu => Device::GPU,
            wgpu::DeviceType::Cpu => Device::CPU,
            wgpu::DeviceType::Other => Device::Auto,
        }
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
