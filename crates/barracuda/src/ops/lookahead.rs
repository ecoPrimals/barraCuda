// SPDX-License-Identifier: AGPL-3.0-only
//! Lookahead - Lookahead Optimizer (Zhang et al.)
//!
//! Maintains two sets of weights: fast and slow.
//! Interpolates between them for better convergence.
//!
//! **Canonical Pattern**: Struct-based with Tensor support

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Lookahead optimizer state.
pub struct LookaheadState {
    /// Slow weights (updated every k steps).
    pub slow_weights: Tensor,
    /// Step counter for sync frequency.
    pub k_counter: usize,
}

/// Lookahead optimizer wrapper
///
/// Maintains fast and slow weight sets, updating slow weights every k steps.
pub struct Lookahead {
    fast_weights: Tensor,
    state: LookaheadState,
    k: usize, // Sync frequency
    alpha: f32,
}

impl Lookahead {
    /// Create a new Lookahead optimizer
    /// ## Parameters
    /// - `fast_weights`: Current fast weights (Tensor)
    /// - `slow_weights`: Slow weights (Tensor, same shape as `fast_weights`)
    /// - `k`: Sync frequency (update slow weights every k steps)
    /// - `alpha`: Slow weights step size (interpolation factor)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    /// Also returns [`Err`] if shape validation fails.
    pub fn new(fast_weights: Tensor, slow_weights: Tensor, k: usize, alpha: f32) -> Result<Self> {
        if fast_weights.len() != slow_weights.len() {
            return Err(BarracudaError::Device(format!(
                "State dimension mismatch: fast_weights len {}, slow_weights len {}",
                fast_weights.len(),
                slow_weights.len()
            )));
        }

        if !std::ptr::eq(
            fast_weights.device().as_ref(),
            slow_weights.device().as_ref(),
        ) {
            return Err(BarracudaError::Device(
                "fast_weights and slow_weights must be on the same device".to_string(),
            ));
        }

        Ok(Self {
            fast_weights,
            state: LookaheadState {
                slow_weights,
                k_counter: 0,
            },
            k,
            alpha,
        })
    }

    /// Execute one Lookahead step
    /// Updates slow weights every k steps, otherwise returns fast weights.
    /// Returns the weights to use for the current step.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(mut self) -> Result<Tensor> {
        self.state.k_counter += 1;

        // Update slow weights every k steps
        if self.state.k_counter.is_multiple_of(self.k) {
            // slow_weights = slow_weights + alpha * (fast_weights - slow_weights)
            // Rearranged: slow_weights = (1 - alpha) * slow_weights + alpha * fast_weights
            //
            // Using tensor ops:
            // 1. diff = fast_weights - slow_weights
            // 2. scaled_diff = alpha * diff
            // 3. new_slow = slow_weights + scaled_diff

            let diff = self.fast_weights.sub(&self.state.slow_weights)?;
            let scaled_diff = diff.mul_scalar(self.alpha)?;
            let new_slow = self.state.slow_weights.add(&scaled_diff)?;

            // Update state for next iteration
            self.state.slow_weights = new_slow.clone();

            Ok(new_slow)
        } else {
            // Return fast weights (no slow weight update this step)
            Ok(self.fast_weights)
        }
    }

    /// Get current state (for checkpointing)
    #[must_use]
    pub fn state(&self) -> &LookaheadState {
        &self.state
    }

    /// Get mutable state (for manual updates)
    pub fn state_mut(&mut self) -> &mut LookaheadState {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lookahead() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let fast_weights = Tensor::from_vec_on(vec![1.0; 100], vec![100], device.clone())
            .await
            .unwrap();
        let slow_weights = Tensor::from_vec_on(vec![0.9; 100], vec![100], device.clone())
            .await
            .unwrap();

        let op = Lookahead::new(fast_weights, slow_weights, 5, 0.5).unwrap();
        let result = op.execute().unwrap();

        assert_eq!(result.len(), 100);
    }
}
