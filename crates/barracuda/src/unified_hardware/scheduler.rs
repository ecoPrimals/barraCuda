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
    pub fn new(executors: Vec<Arc<dyn ComputeExecutor>>) -> Self {
        Self { executors }
    }

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
