// SPDX-License-Identifier: AGPL-3.0-or-later
//! Penalty filtering and surrogate quality utilities.

use super::config::PenaltyFilter;

/// Apply penalty filtering to training data.
///
/// Removes or caps penalty values that would corrupt surrogate training.
pub fn filter_training_data(
    x_data: &[Vec<f64>],
    y_data: &[f64],
    filter: PenaltyFilter,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    match filter {
        PenaltyFilter::None => (x_data.to_vec(), y_data.to_vec()),

        PenaltyFilter::Threshold(threshold) => {
            let (x_filt, y_filt): (Vec<_>, Vec<_>) = x_data
                .iter()
                .zip(y_data.iter())
                .filter(|&(_, y)| *y <= threshold)
                .map(|(x, y)| (x.clone(), *y))
                .unzip();
            (x_filt, y_filt)
        }

        PenaltyFilter::Quantile(q) => {
            if y_data.is_empty() || !(0.0..=1.0).contains(&q) {
                return (x_data.to_vec(), y_data.to_vec());
            }
            let mut sorted: Vec<f64> = y_data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff_idx = ((1.0 - q) * (sorted.len() as f64)).floor() as usize;
            let cutoff_idx = cutoff_idx.min(sorted.len().saturating_sub(1));
            let threshold = sorted[cutoff_idx];

            let (x_filt, y_filt): (Vec<_>, Vec<_>) = x_data
                .iter()
                .zip(y_data.iter())
                .filter(|&(_, y)| *y <= threshold)
                .map(|(x, y)| (x.clone(), *y))
                .unzip();
            (x_filt, y_filt)
        }

        PenaltyFilter::AdaptiveMAD(k) => {
            if y_data.is_empty() {
                return (x_data.to_vec(), y_data.to_vec());
            }
            let mut sorted: Vec<f64> = y_data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if sorted.len().is_multiple_of(2) {
                f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
            } else {
                sorted[sorted.len() / 2]
            };

            let mut deviations: Vec<f64> = y_data.iter().map(|&y| (y - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = if deviations.len().is_multiple_of(2) {
                f64::midpoint(
                    deviations[deviations.len() / 2 - 1],
                    deviations[deviations.len() / 2],
                )
            } else {
                deviations[deviations.len() / 2]
            };

            let threshold = k.mul_add(mad, median);

            let (x_filt, y_filt): (Vec<_>, Vec<_>) = x_data
                .iter()
                .zip(y_data.iter())
                .filter(|&(_, y)| *y <= threshold)
                .map(|(x, y)| (x.clone(), *y))
                .unzip();
            (x_filt, y_filt)
        }
    }
}

/// Compute RMSE of surrogate predictions at training points.
pub(crate) fn compute_surrogate_rmse(
    surrogate: &crate::surrogate::RBFSurrogate,
    x_data: &[Vec<f64>],
    y_data: &[f64],
) -> f64 {
    match surrogate.predict(x_data) {
        Ok(y_pred) => {
            let mse = y_pred
                .iter()
                .zip(y_data.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / y_data.len() as f64;
            mse.sqrt()
        }
        Err(_) => f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_x() -> Vec<Vec<f64>> {
        vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]
    }

    #[test]
    fn filter_none_passes_all() {
        let x = sample_x();
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (xf, yf) = filter_training_data(&x, &y, PenaltyFilter::None);
        assert_eq!(xf.len(), 5);
        assert_eq!(yf, y);
    }

    #[test]
    fn filter_threshold() {
        let x = sample_x();
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (xf, yf) = filter_training_data(&x, &y, PenaltyFilter::Threshold(25.0));
        assert_eq!(yf, vec![10.0, 20.0]);
        assert_eq!(xf.len(), 2);
    }

    #[test]
    fn filter_quantile_keeps_lower() {
        let x = sample_x();
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (filtered_x, yf) = filter_training_data(&x, &y, PenaltyFilter::Quantile(0.5));
        assert!(yf.len() <= 5);
        assert_eq!(filtered_x.len(), yf.len());
        assert!(yf.iter().all(|&v| v <= 30.0));
    }

    #[test]
    fn filter_quantile_empty() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];
        let (xf, yf) = filter_training_data(&x, &y, PenaltyFilter::Quantile(0.5));
        assert!(xf.is_empty());
        assert!(yf.is_empty());
    }

    #[test]
    fn filter_quantile_invalid_range() {
        let x = sample_x();
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (_, yf) = filter_training_data(&x, &y, PenaltyFilter::Quantile(2.0));
        assert_eq!(yf.len(), 5);
    }

    #[test]
    fn filter_adaptive_mad() {
        let x = sample_x();
        let y = vec![10.0, 10.0, 10.0, 10.0, 100.0];
        let (xf, yf) = filter_training_data(&x, &y, PenaltyFilter::AdaptiveMAD(2.0));
        assert!(yf.len() < 5, "outlier should be filtered");
        assert!(xf.len() == yf.len());
    }

    #[test]
    fn filter_adaptive_mad_empty() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];
        let (xf, yf) = filter_training_data(&x, &y, PenaltyFilter::AdaptiveMAD(3.0));
        assert!(xf.is_empty());
        assert!(yf.is_empty());
    }
}
