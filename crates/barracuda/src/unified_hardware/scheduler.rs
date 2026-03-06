// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compute scheduler — selects best hardware for each operation.

use crate::error::Result;
use crate::unified_math::{MathOp, TensorDescriptor};
use std::sync::Arc;

use super::traits::{ComputeExecutor, TensorStorage};

/// Selects best hardware for each operation based on scoring.
pub struct ComputeScheduler {
    executors: Vec<Arc<dyn ComputeExecutor>>,
}

impl ComputeScheduler {
    /// Creates a new scheduler with the given executors.
    #[must_use]
    pub fn new(executors: Vec<Arc<dyn ComputeExecutor>>) -> Self {
        Self { executors }
    }

    /// Selects the best executor for the given operation and inputs based on scoring.
    #[must_use]
    pub fn select_executor(
        &self,
        op: &MathOp,
        inputs: &[TensorDescriptor],
    ) -> Option<Arc<dyn ComputeExecutor>> {
        self.executors
            .iter()
            .filter(|e| e.can_execute(op, inputs))
            .max_by(|a, b| {
                let score_a = a.score_operation(op, inputs);
                let score_b = b.score_operation(op, inputs);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Executes the operation using the best available executor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if no executor can handle the operation, or if execution fails.
    pub async fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> Result<Arc<dyn TensorStorage>> {
        let descriptors: Vec<_> = inputs.iter().map(|t| t.descriptor().clone()).collect();
        let executor = self.select_executor(op, &descriptors).ok_or_else(|| {
            crate::error::BarracudaError::NoAvailableExecutor {
                operation: format!("{op:?}"),
            }
        })?;
        executor.execute(op, inputs).await
    }
}
