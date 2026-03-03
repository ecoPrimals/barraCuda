//! Graph and spectral utilities for linear algebra
//!
//! - [`graph_laplacian`]: Compute graph Laplacian L = D - A from adjacency matrix
//! - [`belief_propagation_chain`]: Chain PGM forward pass (HMM-like)
//! - [`disordered_laplacian`]: Anderson-type diagonal disorder on graph Laplacian
//! - [`effective_rank`]: Effective rank via Shannon entropy of eigenvalue spectrum

/// Compute the graph Laplacian from a flat row-major adjacency matrix.
///
/// L = D - A where D is the degree matrix (diagonal with row sums).
#[must_use]
pub fn graph_laplacian(adjacency: &[f64], n: usize) -> Vec<f64> {
    let mut laplacian = vec![0.0; n * n];
    for i in 0..n {
        let degree: f64 = (0..n).map(|j| adjacency[i * n + j]).sum();
        laplacian[i * n + i] = degree;
        for j in 0..n {
            laplacian[i * n + j] -= adjacency[i * n + j];
        }
    }
    laplacian
}

/// Chain belief propagation: forward pass through a sequence of transition matrices.
///
/// Computes P(layer_k) = normalize(transition_k * P(layer_{k-1})) for each layer.
/// Returns distributions at each layer (including input).
///
/// This is equivalent to the HMM forward algorithm for a chain PGM.
#[must_use]
pub fn belief_propagation_chain(
    input_dist: &[f64],
    transition_matrices: &[&[f64]],
    layer_dims: &[usize],
) -> Vec<Vec<f64>> {
    let mut distributions = Vec::with_capacity(layer_dims.len() + 1);
    distributions.push(input_dist.to_vec());
    let mut current = input_dist.to_vec();
    for (k, trans) in transition_matrices.iter().enumerate() {
        let in_dim = if k == 0 {
            input_dist.len()
        } else {
            layer_dims[k - 1]
        };
        let out_dim = layer_dims[k];
        let mut next = vec![0.0; out_dim];
        for j in 0..out_dim {
            for i in 0..in_dim {
                next[j] += current[i] * trans[i * out_dim + j];
            }
        }
        let sum: f64 = next.iter().sum();
        if sum > 1e-300 {
            for v in &mut next {
                *v /= sum;
            }
        }
        distributions.push(next.clone());
        current = next;
    }
    distributions
}

/// Disordered Laplacian: L + W * diag(heterogeneity - mean)
///
/// Adds Anderson-type diagonal disorder to a graph Laplacian.
/// `heterogeneity[i]` is the disorder value at node i;
/// the mean is subtracted to center the disorder.
#[must_use]
pub fn disordered_laplacian(
    laplacian: &[f64],
    n: usize,
    heterogeneity: &[f64],
    disorder_strength: f64,
) -> Vec<f64> {
    let mean: f64 = heterogeneity.iter().sum::<f64>() / n as f64;
    let mut result = laplacian.to_vec();
    for i in 0..n {
        result[i * n + i] += disorder_strength * (heterogeneity[i] - mean);
    }
    result
}

/// Effective rank via Shannon entropy of normalized eigenvalue spectrum.
///
/// rank_eff = exp(H) where H = -sum(p_i * log(p_i)).
#[must_use]
pub fn effective_rank(eigenvalues: &[f64]) -> f64 {
    let abs_vals: Vec<f64> = eigenvalues.iter().map(|&ev| ev.abs()).collect();
    let total: f64 = abs_vals.iter().sum();
    if total < 1e-300 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &v in &abs_vals {
        let p = v / total;
        if p > 1e-300 {
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_laplacian_row_sums_zero() {
        // 3-node path: 0--1--2
        let adj = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let l = graph_laplacian(&adj, 3);
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| l[i * 3 + j]).sum();
            assert!(
                row_sum.abs() < 1e-14,
                "row {} sum = {}, expected 0",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_graph_laplacian_identity() {
        // 2x2 complete graph: both nodes connected
        // A = [[0,1],[1,0]] -> D = [[1,0],[0,1]], L = D - A = [[1,-1],[-1,1]]
        let adj = vec![0.0, 1.0, 1.0, 0.0];
        let l = graph_laplacian(&adj, 2);
        assert!((l[0] - 1.0).abs() < 1e-14);
        assert!((l[1] - (-1.0)).abs() < 1e-14);
        assert!((l[2] - (-1.0)).abs() < 1e-14);
        assert!((l[3] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_graph_laplacian_empty() {
        // Zero adjacency (disconnected graph)
        let adj = vec![0.0, 0.0, 0.0, 0.0];
        let l = graph_laplacian(&adj, 2);
        assert_eq!(l, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_effective_rank_full() {
        // Equal eigenvalues -> full rank (entropy max, rank_eff = n)
        let ev = vec![1.0, 1.0, 1.0, 1.0];
        let r = effective_rank(&ev);
        assert!((r - 4.0).abs() < 1e-10, "expected 4, got {}", r);
    }

    #[test]
    fn test_effective_rank_single() {
        // One nonzero -> rank 1
        let ev = vec![1.0, 0.0, 0.0];
        let r = effective_rank(&ev);
        assert!((r - 1.0).abs() < 1e-10, "expected 1, got {}", r);
    }

    #[test]
    fn test_effective_rank_zero() {
        // All zeros -> 0
        let ev = vec![0.0, 0.0, 0.0];
        let r = effective_rank(&ev);
        assert!((r - 0.0).abs() < 1e-14, "expected 0, got {}", r);
    }

    #[test]
    fn test_belief_propagation_identity() {
        // Identity transition preserves distribution
        let input = vec![0.5, 0.3, 0.2];
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let trans = [identity.as_slice()];
        let layer_dims = [3];
        let dists = belief_propagation_chain(&input, &trans, &layer_dims);
        assert_eq!(dists.len(), 2);
        for (a, b) in dists[1].iter().zip(input.iter()) {
            assert!(
                (a - b).abs() < 1e-14,
                "identity should preserve: {:?} vs {:?}",
                dists[1],
                input
            );
        }
    }

    #[test]
    fn test_belief_propagation_two_layers() {
        // 2-layer chain produces valid distributions
        let input = vec![1.0, 0.0]; // deterministic at state 0
        let t1 = vec![0.5, 0.5, 0.5, 0.5]; // 2x2 uniform-ish
        let t2 = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let trans = [t1.as_slice(), t2.as_slice()];
        let layer_dims = [2, 2];
        let dists = belief_propagation_chain(&input, &trans, &layer_dims);
        assert_eq!(dists.len(), 3);
        for d in &dists {
            let sum: f64 = d.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "distribution should sum to 1: {:?} sum={}",
                d,
                sum
            );
        }
    }

    #[test]
    fn test_belief_propagation_normalization() {
        // Output distributions sum to 1
        let input = vec![0.25, 0.25, 0.25, 0.25];
        let t = vec![0.25; 16]; // 4x4 uniform
        let trans = [t.as_slice()];
        let layer_dims = [4];
        let dists = belief_propagation_chain(&input, &trans, &layer_dims);
        for d in &dists {
            let sum: f64 = d.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "distribution should sum to 1: {:?} sum={}",
                d,
                sum
            );
        }
    }

    #[test]
    fn test_disordered_laplacian_zero_strength() {
        // Zero disorder strength -> equals input laplacian
        let l = vec![1.0, -1.0, -1.0, 1.0];
        let h = vec![0.5, 1.5];
        let result = disordered_laplacian(&l, 2, &h, 0.0);
        assert_eq!(result, l);
    }

    #[test]
    fn test_disordered_laplacian_symmetry() {
        // Disorder only affects diagonal; off-diagonals unchanged
        let l = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let h = vec![0.1, 0.2, 0.3];
        let result = disordered_laplacian(&l, 3, &h, 1.0);
        assert!((result[1] - l[1]).abs() < 1e-14);
        assert!((result[2] - l[2]).abs() < 1e-14);
        assert!((result[3] - l[3]).abs() < 1e-14);
        assert!((result[5] - l[5]).abs() < 1e-14);
        assert!((result[6] - l[6]).abs() < 1e-14);
        assert!((result[7] - l[7]).abs() < 1e-14);
    }

    #[test]
    fn test_disordered_laplacian_centered() {
        // Sum of disorder terms (diagonal additions) is zero
        let l = vec![1.0, 0.0, 0.0, 1.0];
        let h = vec![1.0, 2.0, 3.0]; // mean = 2
        let result = disordered_laplacian(&l, 2, &h[..2], 1.0);
        // h[0]-mean=-1, h[1]-mean=0; diagonal adds: -1 and 0
        // Original diag: 1, 1 -> result: 0, 1
        let diag_adds: Vec<f64> = (0..2).map(|i| result[i * 2 + i] - l[i * 2 + i]).collect();
        let sum_adds: f64 = diag_adds.iter().sum();
        assert!(
            sum_adds.abs() < 1e-14,
            "centered disorder: diagonal additions should sum to 0, got {}",
            sum_adds
        );
    }
}
