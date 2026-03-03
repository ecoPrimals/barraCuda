//! Loss Functions
//!
//! Loss function implementations for training neural networks.

/// Loss function types
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// Cross entropy loss
    CrossEntropy,
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
}
