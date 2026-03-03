//! Training and Evaluation Metrics
//!
//! Runtime metrics collection for neural network training.
//! Deep Debt compliant: No hardcoded thresholds, all runtime data.

/// Training metrics (runtime data)
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub accuracy: Option<f32>,
    pub epoch: usize,
    pub batch: usize,
}

/// Training history (runtime accumulation)
#[derive(Debug, Clone, Default)]
pub struct TrainHistory {
    pub losses: Vec<f32>,
    pub accuracies: Vec<f32>,
    pub epochs_completed: usize,
}

/// Evaluation metrics
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub samples: usize,
}
