// SPDX-License-Identifier: AGPL-3.0-or-later
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_variants_debug() {
        assert_eq!(format!("{:?}", LossFunction::CrossEntropy), "CrossEntropy");
        assert_eq!(format!("{:?}", LossFunction::MSE), "MSE");
        assert_eq!(format!("{:?}", LossFunction::MAE), "MAE");
    }

    #[test]
    fn loss_clone() {
        let loss = LossFunction::MSE;
        #[expect(
            clippy::redundant_clone,
            reason = "clone needed for borrowck across async boundary"
        )]
        let cloned = loss.clone();
        assert!(matches!(cloned, LossFunction::MSE));
    }
}
