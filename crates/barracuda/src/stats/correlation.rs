//! Correlation and covariance functions
//!
//! # Functions
//!
//! - `pearson_correlation`: Pearson product-moment correlation coefficient
//! - `covariance`: Sample covariance between two variables
//! - `correlation_matrix`: Pairwise correlations for multiple variables
//! - `covariance_matrix`: Pairwise covariances for multiple variables
//!
//! # Notes
//!
//! All functions use sample statistics (dividing by n-1 for covariance).

use crate::error::{BarracudaError, Result};

/// Compute Pearson correlation coefficient between two vectors.
///
/// r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² Σ(yᵢ - ȳ)²]
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable (must have same length as x)
///
/// # Returns
///
/// Correlation coefficient in [-1, 1], or NaN if either variance is zero.
///
/// # Examples
///
/// ```
/// use barracuda::stats::pearson_correlation;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let r = pearson_correlation(&x, &y).unwrap();
/// assert!((r - 1.0).abs() < 1e-10);  // Perfect positive correlation
/// ```
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!("Vectors must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "Need at least 2 data points for correlation".to_string(),
        });
    }

    let n = x.len() as f64;

    // Compute means
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    // Compute covariance and variances in one pass
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        // One or both variables have zero variance
        return Ok(f64::NAN);
    }

    Ok(cov / denom)
}

/// Compute sample covariance between two vectors.
///
/// Cov(X,Y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n-1)
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable (must have same length as x)
///
/// # Examples
///
/// ```
/// use barracuda::stats::covariance;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let cov = covariance(&x, &y).unwrap();
/// assert!((cov - 5.0).abs() < 1e-10);
/// ```
pub fn covariance(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!("Vectors must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "Need at least 2 data points for covariance".to_string(),
        });
    }

    let n = x.len() as f64;

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    // Sample covariance: divide by n-1
    Ok(cov / (n - 1.0))
}

/// Compute sample variance of a vector.
///
/// Var(X) = Σ(xᵢ - x̄)² / (n-1)
pub fn variance(x: &[f64]) -> Result<f64> {
    if x.len() < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "Need at least 2 data points for variance".to_string(),
        });
    }

    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;

    let var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum();

    Ok(var / (n - 1.0))
}

/// Compute standard deviation of a vector.
pub fn std_dev(x: &[f64]) -> Result<f64> {
    Ok(variance(x)?.sqrt())
}

/// Compute correlation matrix for multiple variables.
///
/// Given data matrix with n observations (rows) and p variables (columns),
/// returns a p×p symmetric matrix where element (i,j) is the correlation
/// between variable i and variable j.
///
/// # Arguments
///
/// * `data` - Data matrix as row-major Vec of rows, each row is one observation
///
/// # Returns
///
/// p×p correlation matrix (flattened row-major)
///
/// # Examples
///
/// ```
/// use barracuda::stats::correlation_matrix;
///
/// // 5 observations, 2 variables
/// let data = vec![
///     vec![1.0, 2.0],
///     vec![2.0, 4.0],
///     vec![3.0, 6.0],
///     vec![4.0, 8.0],
///     vec![5.0, 10.0],
/// ];
/// let corr = correlation_matrix(&data).unwrap();
/// // corr is 2x2: [[1, r], [r, 1]] where r ≈ 1.0
/// assert!((corr[0] - 1.0).abs() < 1e-10);  // var1 with var1
/// assert!((corr[1] - 1.0).abs() < 1e-10);  // var1 with var2 (perfect correlation)
/// ```
pub fn correlation_matrix(data: &[Vec<f64>]) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "Data matrix is empty".to_string(),
        });
    }

    let n = data.len();
    if n < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "Need at least 2 observations".to_string(),
        });
    }

    let p = data[0].len();
    if p == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "Data has no variables".to_string(),
        });
    }

    // Verify all rows have same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != p {
            return Err(BarracudaError::InvalidInput {
                message: format!("Row {} has {} elements, expected {}", i, row.len(), p),
            });
        }
    }

    // Extract columns
    let columns: Vec<Vec<f64>> = (0..p)
        .map(|j| data.iter().map(|row| row[j]).collect())
        .collect();

    // Compute correlation matrix
    let mut corr = vec![0.0; p * p];
    for i in 0..p {
        corr[i * p + i] = 1.0; // Diagonal is always 1
        for j in (i + 1)..p {
            let r = pearson_correlation(&columns[i], &columns[j])?;
            corr[i * p + j] = r;
            corr[j * p + i] = r; // Symmetric
        }
    }

    Ok(corr)
}

/// Compute covariance matrix for multiple variables.
///
/// Given data matrix with n observations (rows) and p variables (columns),
/// returns a p×p symmetric matrix where element (i,j) is the covariance
/// between variable i and variable j.
///
/// # Arguments
///
/// * `data` - Data matrix as row-major Vec of rows
///
/// # Returns
///
/// p×p covariance matrix (flattened row-major)
pub fn covariance_matrix(data: &[Vec<f64>]) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "Data matrix is empty".to_string(),
        });
    }

    let n = data.len();
    if n < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "Need at least 2 observations".to_string(),
        });
    }

    let p = data[0].len();
    if p == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "Data has no variables".to_string(),
        });
    }

    // Verify all rows have same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != p {
            return Err(BarracudaError::InvalidInput {
                message: format!("Row {} has {} elements, expected {}", i, row.len(), p),
            });
        }
    }

    // Extract columns
    let columns: Vec<Vec<f64>> = (0..p)
        .map(|j| data.iter().map(|row| row[j]).collect())
        .collect();

    // Compute covariance matrix
    let mut cov_mat = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            let c = covariance(&columns[i], &columns[j])?;
            cov_mat[i * p + j] = c;
            cov_mat[j * p + i] = c; // Symmetric
        }
    }

    Ok(cov_mat)
}

/// Compute Spearman rank correlation coefficient.
///
/// Measures monotonic relationship (not just linear).
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!("Vectors must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "Need at least 2 data points".to_string(),
        });
    }

    // Compute ranks
    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    // Pearson correlation of ranks
    pearson_correlation(&rank_x, &rank_y)
}

/// Helper: compute ranks of values (1-based, average for ties).
fn compute_ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut indexed: Vec<(usize, f64)> = x.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all tied values
        while j < n - 1 && indexed[j].1 == indexed[j + 1].1 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!((r + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_zero() {
        // Orthogonal data
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn test_pearson_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(pearson_correlation(&x, &y).is_err());
    }

    #[test]
    fn test_covariance_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let cov = covariance(&x, &y).unwrap();
        // Cov(X, 2X) = 2 Var(X) = 2 * 2.5 = 5.0
        assert!((cov - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = variance(&x).unwrap();
        // Var = 2.5 (sample variance)
        assert!((v - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix() {
        // 5 observations, 2 perfectly correlated variables
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
        ];
        let corr = correlation_matrix(&data).unwrap();

        // 2x2 matrix, flattened
        assert_eq!(corr.len(), 4);
        assert!((corr[0] - 1.0).abs() < 1e-10); // [0,0]
        assert!((corr[1] - 1.0).abs() < 1e-10); // [0,1]
        assert!((corr[2] - 1.0).abs() < 1e-10); // [1,0]
        assert!((corr[3] - 1.0).abs() < 1e-10); // [1,1]
    }

    #[test]
    fn test_covariance_matrix() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
        ];
        let cov = covariance_matrix(&data).unwrap();

        // Var(X) = 2.5, Var(Y) = 10.0, Cov(X,Y) = 5.0
        assert!((cov[0] - 2.5).abs() < 1e-10); // [0,0]
        assert!((cov[1] - 5.0).abs() < 1e-10); // [0,1]
        assert!((cov[2] - 5.0).abs() < 1e-10); // [1,0]
        assert!((cov[3] - 10.0).abs() < 1e-10); // [1,1]
    }

    #[test]
    fn test_spearman_monotonic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x², monotonic
        let r = spearman_correlation(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_with_ties() {
        let x = vec![1.0, 2.0, 2.0, 4.0];
        let y = vec![1.0, 3.0, 2.0, 4.0];
        let r = spearman_correlation(&x, &y).unwrap();
        // Should still work with ties
        assert!(r > 0.5);
    }

    #[test]
    fn test_compute_ranks() {
        let x = vec![3.0, 1.0, 2.0];
        let ranks = compute_ranks(&x);
        assert!((ranks[0] - 3.0).abs() < 1e-10); // 3.0 is largest
        assert!((ranks[1] - 1.0).abs() < 1e-10); // 1.0 is smallest
        assert!((ranks[2] - 2.0).abs() < 1e-10); // 2.0 is middle
    }

    #[test]
    fn test_compute_ranks_with_ties() {
        let x = vec![1.0, 2.0, 2.0, 4.0];
        let ranks = compute_ranks(&x);
        assert!((ranks[0] - 1.0).abs() < 1e-10);
        assert!((ranks[1] - 2.5).abs() < 1e-10); // Tied values get average rank
        assert!((ranks[2] - 2.5).abs() < 1e-10);
        assert!((ranks[3] - 4.0).abs() < 1e-10);
    }
}
