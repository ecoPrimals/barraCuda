// SPDX-License-Identifier: AGPL-3.0-only
//! Training and Evaluation Metrics
//!
//! Runtime metrics collection for neural network training.
//! Deep Debt compliant: No hardcoded thresholds, all runtime data.

/// Training metrics (runtime data).
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Loss value
    pub loss: f32,
    /// Accuracy (if applicable)
    pub accuracy: Option<f32>,
    /// Current epoch
    pub epoch: usize,
    /// Current batch index
    pub batch: usize,
}

/// Training history (runtime accumulation).
#[derive(Debug, Clone, Default)]
pub struct TrainHistory {
    /// Loss per step/batch
    pub losses: Vec<f32>,
    /// Accuracy per step/batch
    pub accuracies: Vec<f32>,
    /// Number of epochs completed
    pub epochs_completed: usize,
}

/// Evaluation metrics.
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    /// Loss on evaluation set
    pub loss: f32,
    /// Accuracy on evaluation set
    pub accuracy: f32,
    /// Number of samples evaluated
    pub samples: usize,
}
