// SPDX-License-Identifier: AGPL-3.0-or-later
//! `OneCycle` - 1cycle Learning Rate Policy (Smith)
//!
//! Single cycle: warmup to `max_lr`, then anneal to `min_lr`.
//! Enables super-convergence with high learning rates.
//!
//! **Canonical Pattern**: Struct-based with Tensor support

use crate::error::Result;
use crate::tensor::Tensor;

/// `OneCycle` learning rate scheduler
///
/// Implements the 1cycle policy: warmup to `max_lr`, then anneal to `min_lr`.
pub struct OneCycle {
    max_lr: f32,
    total_steps: usize,
    current_step: usize,
    pct_start: f32,        // Percentage of cycle spent warming up (default 0.3)
    div_factor: f32,       // Initial lr = max_lr / div_factor
    final_div_factor: f32, // Final lr = max_lr / final_div_factor
}

impl OneCycle {
    /// Create a new `OneCycle` learning rate scheduler
    /// ## Parameters
    /// - `max_lr`: Maximum learning rate (peak)
    /// - `total_steps`: Total number of training steps
    /// - `current_step`: Current step (0-indexed)
    /// - `pct_start`: Percentage of cycle spent warming up (default 0.3)
    /// - `div_factor`: Initial lr = `max_lr` / `div_factor` (default 25.0)
    /// - `final_div_factor`: Final lr = `max_lr` / `final_div_factor` (default 10000.0)
    /// # Errors
    /// Returns [`Err`] if `current_step` >= `total_steps`.
    pub fn new(
        max_lr: f32,
        total_steps: usize,
        current_step: usize,
        pct_start: f32,
        div_factor: f32,
        final_div_factor: f32,
    ) -> Result<Self> {
        if current_step >= total_steps {
            return Err(crate::error::BarracudaError::Device(format!(
                "Current step {current_step} exceeds total steps {total_steps}"
            )));
        }

        Ok(Self {
            max_lr,
            total_steps,
            current_step,
            pct_start,
            div_factor,
            final_div_factor,
        })
    }

    /// Execute `OneCycle` learning rate calculation
    /// Returns the learning rate for the current step.
    /// This is a CPU calculation (learning rate scheduling doesn't need GPU).
    /// # Errors
    /// This function does not return errors; the [`Result`] type is for API consistency.
    pub fn execute(self) -> Result<f32> {
        let step = self.current_step as f32;
        let total = self.total_steps as f32;
        let warmup_steps = (self.pct_start * total).floor();

        let lr = if step < warmup_steps {
            // Warmup phase: increase from initial_lr to max_lr
            let initial_lr = self.max_lr / self.div_factor;
            let pct = step / warmup_steps;
            (self.max_lr - initial_lr).mul_add(pct, initial_lr)
        } else {
            // Annealing phase: decrease from max_lr to final_lr
            let final_lr = self.max_lr / self.final_div_factor;
            let pct = (step - warmup_steps) / (total - warmup_steps);
            (self.max_lr - final_lr).mul_add(-pct, self.max_lr)
        };

        Ok(lr)
    }

    /// Get learning rate as a scalar tensor
    /// Creates a tensor with a single element containing the learning rate.
    /// Useful for operations that require tensor inputs.
    /// Note: This requires async context. For synchronous use, use `execute()` and create tensor manually.
    /// # Errors
    /// Returns [`Err`] if `execute()` fails or device creation fails.
    pub async fn execute_as_tensor(self) -> Result<Tensor> {
        let lr = self.execute()?;
        let device = crate::device::Auto::new().await?;
        Ok(Tensor::new(vec![lr], vec![1], device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onecycle() {
        let op = OneCycle::new(0.01, 10_000, 0, 0.3, 25.0, 10_000.0).unwrap();
        let lr_start = op.execute().unwrap();

        let op = OneCycle::new(0.01, 10_000, 3000, 0.3, 25.0, 10_000.0).unwrap();
        let lr_peak = op.execute().unwrap();

        let op = OneCycle::new(0.01, 10_000, 9999, 0.3, 25.0, 10_000.0).unwrap();
        let lr_end = op.execute().unwrap();

        assert!(lr_peak > lr_start);
        assert!(lr_end < lr_peak);
    }
}
