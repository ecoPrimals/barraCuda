// SPDX-License-Identifier: AGPL-3.0-or-later
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_metrics_construction() {
        let m = TrainingMetrics {
            loss: 0.5,
            accuracy: Some(0.95),
            epoch: 10,
            batch: 32,
        };
        assert!((m.loss - 0.5).abs() < 1e-6);
        assert_eq!(m.accuracy, Some(0.95));
        assert_eq!(m.epoch, 10);
    }

    #[test]
    fn train_history_default() {
        let h = TrainHistory::default();
        assert!(h.losses.is_empty());
        assert!(h.accuracies.is_empty());
        assert_eq!(h.epochs_completed, 0);
    }

    #[test]
    fn train_history_accumulation() {
        let mut h = TrainHistory::default();
        h.losses.push(1.0);
        h.losses.push(0.5);
        h.epochs_completed = 2;
        assert_eq!(h.losses.len(), 2);
    }

    #[test]
    fn eval_metrics_construction() {
        let m = EvalMetrics {
            loss: 0.1,
            accuracy: 0.99,
            samples: 10_000,
        };
        assert_eq!(m.samples, 10_000);
    }
}
