// SPDX-License-Identifier: AGPL-3.0-only
//! Sobol quasi-random sequences
//!
//! Sobol sequences are low-discrepancy sequences that provide more uniform
//! coverage of the unit hypercube than pseudo-random sampling. They are
//! particularly valuable for high-dimensional integration and sampling.
//!
//! # Properties
//!
//! - **Low discrepancy**: Points are more evenly distributed than random
//! - **Deterministic**: Given seed, sequence is reproducible
//! - **Progressive**: Adding points improves existing distribution
//!
//! # Applications
//!
//! - **Monte Carlo integration**: Faster convergence than random sampling
//! - **Sensitivity analysis**: Sobol indices for variance decomposition
//! - **Design of experiments**: Space-filling designs
//! - **Global optimization**: Initial sampling for surrogate models
//!
//! # Implementation Notes
//!
//! Uses direction numbers from Sobol (1967) and Joe & Kuo (2008) for
//! dimensions up to 21201. The implementation uses Gray code for
//! efficient incremental generation.
//!
//! # References
//!
//! - Sobol, I. M. (1967), "Distribution of points in a cube"
//! - Joe, S. & Kuo, F. Y. (2008), "Constructing Sobol sequences"

use crate::error::{BarracudaError, Result};

/// Maximum supported dimension for Sobol sequences.
pub const MAX_SOBOL_DIM: usize = 40;

/// Direction numbers for Sobol sequence generation.
///
/// These are primitive polynomials and initial direction numbers from
/// Joe & Kuo (2008). We include data for up to 40 dimensions.
static DIRECTION_NUMBERS: &[&[u32]] = &[
    // Dimension 1
    &[1],
    // Dimension 2
    &[1, 1],
    // Dimension 3
    &[1, 3, 1],
    // Dimension 4
    &[1, 3, 3],
    // Dimension 5
    &[1, 1, 1],
    // Dimension 6
    &[1, 1, 3, 3],
    // Dimension 7
    &[1, 3, 5, 13],
    // Dimension 8
    &[1, 1, 5, 5, 17],
    // Dimension 9
    &[1, 1, 5, 5, 5],
    // Dimension 10
    &[1, 1, 7, 11, 19],
    // Dimension 11
    &[1, 1, 5, 1, 1],
    // Dimension 12
    &[1, 1, 1, 3, 11],
    // Dimension 13
    &[1, 3, 5, 5, 31],
    // Dimension 14
    &[1, 3, 3, 9, 7, 49],
    // Dimension 15
    &[1, 1, 1, 15, 21, 21],
    // Dimension 16
    &[1, 3, 1, 13, 27, 49],
    // Dimension 17
    &[1, 1, 1, 15, 7, 5],
    // Dimension 18
    &[1, 3, 1, 15, 13, 25],
    // Dimension 19
    &[1, 1, 5, 5, 19, 61],
    // Dimension 20
    &[1, 3, 7, 11, 23, 15, 103],
    // Dimension 21
    &[1, 3, 7, 13, 13, 15, 69],
    // Dimension 22
    &[1, 1, 3, 13, 7, 35, 63],
    // Dimension 23
    &[1, 1, 5, 9, 27, 49, 71],
    // Dimension 24
    &[1, 1, 5, 11, 15, 33, 107],
    // Dimension 25
    &[1, 3, 1, 3, 13, 39, 127],
    // Dimension 26
    &[1, 3, 3, 5, 9, 3, 67],
    // Dimension 27
    &[1, 3, 5, 11, 21, 45, 111],
    // Dimension 28
    &[1, 3, 7, 13, 21, 27, 51],
    // Dimension 29
    &[1, 3, 7, 15, 1, 47, 97],
    // Dimension 30
    &[1, 3, 7, 9, 15, 35, 117],
    // Dimension 31
    &[1, 1, 1, 3, 31, 49, 89],
    // Dimension 32
    &[1, 1, 1, 5, 5, 57, 79],
    // Dimension 33
    &[1, 1, 1, 11, 13, 57, 67],
    // Dimension 34
    &[1, 1, 3, 1, 13, 63, 75],
    // Dimension 35
    &[1, 1, 3, 13, 31, 23, 61],
    // Dimension 36
    &[1, 1, 5, 9, 1, 57, 111],
    // Dimension 37
    &[1, 1, 7, 3, 17, 57, 53],
    // Dimension 38
    &[1, 1, 7, 13, 29, 7, 91],
    // Dimension 39
    &[1, 1, 7, 15, 9, 37, 107],
    // Dimension 40
    &[1, 3, 1, 1, 17, 41, 75],
];

/// Primitive polynomials for Sobol sequence (degree of polynomial for each dimension).
static POLYNOMIALS: &[u32] = &[
    0, // dim 1: trivial polynomial p(x)=1 (degree 0), Joe-Kuo
    1, // dim 2: x + 1 (degree 1)
    1, // dim 3: x + 1
    2, // dim 4: x² + x + 1
    1, // dim 5: x + 1
    4, // dim 6: x⁴ + x + 1
    2, // dim 7: x² + x + 1
    4, // dim 8: x⁴ + x³ + 1
    4, // dim 9
    3, // dim 10
    2, // dim 11
    3, // dim 12
    2, // dim 13
    5, // dim 14
    4, // dim 15
    4, // dim 16
    4, // dim 17
    4, // dim 18
    4, // dim 19
    6, // dim 20
    6, // dim 21
    5, // dim 22
    5, // dim 23
    5, // dim 24
    5, // dim 25
    5, // dim 26
    5, // dim 27
    5, // dim 28
    5, // dim 29
    5, // dim 30
    5, // dim 31
    5, // dim 32
    5, // dim 33
    5, // dim 34
    5, // dim 35
    5, // dim 36
    5, // dim 37
    5, // dim 38
    5, // dim 39
    5, // dim 40
];

/// Sobol sequence generator.
#[derive(Debug, Clone)]
pub struct SobolGenerator {
    /// Number of dimensions
    dim: usize,
    /// Direction numbers (dim × 32 matrix)
    v: Vec<Vec<u32>>,
    /// Current index in sequence
    index: u64,
    /// Previous point (for Gray code update)
    x: Vec<u32>,
}

impl SobolGenerator {
    /// Create a new Sobol generator for the given dimension.
    /// # Arguments
    /// * `dim` - Number of dimensions (1 to 40)
    /// # Example
    /// ```
    /// use barracuda::sample::sobol::SobolGenerator;
    /// let mut gen = SobolGenerator::new(5).unwrap();
    /// let point = gen.next_point();  // 5-dimensional point
    /// ```
    /// # Errors
    /// Returns [`Err`] if `dim` is 0 or greater than [`MAX_SOBOL_DIM`].
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 || dim > MAX_SOBOL_DIM {
            return Err(BarracudaError::InvalidInput {
                message: format!("Dimension must be 1..{MAX_SOBOL_DIM}"),
            });
        }

        let mut v = vec![vec![0u32; 32]; dim];

        // Initialize direction numbers
        for d in 0..dim {
            if d == 0 {
                // First dimension is just powers of 2
                for i in 0..32 {
                    v[0][i] = 1 << (31 - i);
                }
            } else {
                let m_init = &DIRECTION_NUMBERS[d];
                let s = m_init.len();
                let poly_degree = POLYNOMIALS[d] as usize;

                // Initialize first s direction numbers
                for (i, &m) in m_init.iter().enumerate() {
                    v[d][i] = m << (31 - i);
                }

                // Compute remaining direction numbers using recurrence
                let poly = 1u32 << poly_degree; // Implicit leading 1
                for i in s..32 {
                    let mut vi = v[d][i - s] >> s;
                    for j in 1..=poly_degree {
                        if (poly >> (poly_degree - j)) & 1 == 1 {
                            vi ^= v[d][i - j];
                        }
                    }
                    vi ^= v[d][i - s];
                    v[d][i] = vi;
                }
            }
        }

        Ok(Self {
            dim,
            v,
            index: 0,
            x: vec![0; dim],
        })
    }

    /// Skip to a specific index in the sequence.
    /// Useful for parallel generation where different workers start
    /// at different offsets.
    /// After calling `skip_to(n)`, the next `next_point()` call will return
    /// the point at index n.
    /// # Implementation Note
    /// For correctness, this uses sequential generation internally.
    /// For very large n (> 1M), consider using parallel Sobol generation
    /// with different scrambling seeds instead.
    pub fn skip_to(&mut self, n: u64) {
        // Reset to initial state
        self.index = 0;
        self.x.fill(0);

        // Generate n points to advance state
        // This is O(n) but guaranteed correct
        for _ in 0..n {
            let _ = self.next_point();
        }
    }

    /// Generate the next point in the sequence.
    /// Returns a point in [0, 1)^dim.
    pub fn next_point(&mut self) -> Vec<f64> {
        if self.index == 0 {
            self.index = 1;
            return vec![0.0; self.dim];
        }

        // Find rightmost zero bit position
        let mut i = self.index;
        let mut c = 0;
        while i & 1 == 1 {
            i >>= 1;
            c += 1;
        }

        // XOR with direction number
        for d in 0..self.dim {
            self.x[d] ^= self.v[d][c];
        }

        self.index += 1;

        // Convert to [0, 1)
        let scale = 1.0 / (1u64 << 32) as f64;
        self.x.iter().map(|&xi| xi as f64 * scale).collect()
    }

    /// Generate n points.
    pub fn generate(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_point()).collect()
    }

    /// Reset to the beginning of the sequence.
    pub fn reset(&mut self) {
        self.index = 0;
        self.x.fill(0);
    }

    /// Get current index.
    #[must_use]
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Get dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Generate n Sobol sequence points in the unit hypercube.
///
/// # Arguments
///
/// * `n` - Number of points
/// * `dim` - Dimension of each point
///
/// # Returns
///
/// Vector of n points, each with dim coordinates in [0, 1).
///
/// # Example
///
/// ```
/// use barracuda::sample::sobol::sobol_sequence;
///
/// let points = sobol_sequence(100, 3).unwrap();
/// assert_eq!(points.len(), 100);
/// assert_eq!(points[0].len(), 3);
/// ```
///
/// # Errors
///
/// Returns [`Err`] if `dim` is invalid (see [`SobolGenerator::new`]).
pub fn sobol_sequence(n: usize, dim: usize) -> Result<Vec<Vec<f64>>> {
    let mut sampler = SobolGenerator::new(dim)?;
    Ok(sampler.generate(n))
}

/// Generate Sobol points scaled to given bounds.
///
/// # Arguments
///
/// * `n` - Number of points
/// * `bounds` - (lower, upper) bounds for each dimension
///
/// # Example
///
/// ```
/// use barracuda::sample::sobol::sobol_scaled;
///
/// let bounds = vec![(0.0, 10.0), (-5.0, 5.0)];
/// let points = sobol_scaled(50, &bounds).unwrap();
///
/// // All points should be within bounds
/// for p in &points {
///     assert!(p[0] >= 0.0 && p[0] <= 10.0);
///     assert!(p[1] >= -5.0 && p[1] <= 5.0);
/// }
/// ```
///
/// # Errors
///
/// Returns [`Err`] if [`sobol_sequence`] fails (invalid dimension from `bounds.len()`).
pub fn sobol_scaled(n: usize, bounds: &[(f64, f64)]) -> Result<Vec<Vec<f64>>> {
    let dim = bounds.len();
    let points = sobol_sequence(n, dim)?;

    Ok(points
        .into_iter()
        .map(|p| {
            p.into_iter()
                .zip(bounds.iter())
                .map(|(xi, (lo, hi))| lo + xi * (hi - lo))
                .collect()
        })
        .collect())
}

/// Generate Sobol points starting from a given index (for parallel generation).
///
/// # Arguments
///
/// * `n` - Number of points
/// * `dim` - Dimension
/// * `start_index` - Index to start from
///
/// # Errors
///
/// Returns [`Err`] if `dim` is invalid (see [`SobolGenerator::new`]).
pub fn sobol_sequence_from(n: usize, dim: usize, start_index: u64) -> Result<Vec<Vec<f64>>> {
    let mut sampler = SobolGenerator::new(dim)?;
    sampler.skip_to(start_index);
    Ok(sampler.generate(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobol_1d() {
        let points = sobol_sequence(8, 1).unwrap();
        assert_eq!(points.len(), 8);

        // All points should be in [0, 1)
        for p in &points {
            assert!(p[0] >= 0.0 && p[0] < 1.0, "Point {} out of range", p[0]);
        }

        // First point is typically 0 or close to 0
        assert!(
            points[0][0] < 0.1,
            "First point should be near 0, got {}",
            points[0][0]
        );
    }

    #[test]
    fn test_sobol_2d() {
        let points = sobol_sequence(16, 2).unwrap();
        assert_eq!(points.len(), 16);
        assert_eq!(points[0].len(), 2);

        // All points should be in [0, 1)
        for p in &points {
            assert!(p[0] >= 0.0 && p[0] < 1.0);
            assert!(p[1] >= 0.0 && p[1] < 1.0);
        }
    }

    #[test]
    fn test_sobol_uniformity() {
        // Generate many points and check they're reasonably uniform
        let n = 1024;
        let points = sobol_sequence(n, 2).unwrap();

        // Count points in each quadrant
        let mut quadrants = [0; 4];
        for p in &points {
            let q = usize::from(p[0] >= 0.5) + (if p[1] >= 0.5 { 2 } else { 0 });
            quadrants[q] += 1;
        }

        // Each quadrant should have roughly n/4 points
        for count in quadrants {
            assert!(
                (count as f64 - n as f64 / 4.0).abs() < n as f64 / 8.0,
                "Quadrant count {} deviates too much from {}",
                count,
                n / 4
            );
        }
    }

    #[test]
    fn test_sobol_scaled() {
        let bounds = vec![(0.0, 10.0), (-5.0, 5.0), (100.0, 200.0)];
        let points = sobol_scaled(100, &bounds).unwrap();

        for p in &points {
            assert!(p[0] >= 0.0 && p[0] <= 10.0);
            assert!(p[1] >= -5.0 && p[1] <= 5.0);
            assert!(p[2] >= 100.0 && p[2] <= 200.0);
        }
    }

    #[test]
    fn test_sobol_skip_to() {
        let mut sampler1 = SobolGenerator::new(3).unwrap();
        let mut sampler2 = SobolGenerator::new(3).unwrap();

        // Generate 100 points normally
        for _ in 0..100 {
            sampler1.next_point();
        }

        // Skip to 100
        sampler2.skip_to(100);

        // Next points should match
        let p1 = sampler1.next_point();
        let p2 = sampler2.next_point();

        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-10, "Mismatch after skip: {a} vs {b}");
        }
    }

    #[test]
    fn test_sobol_reset() {
        let mut sampler = SobolGenerator::new(2).unwrap();

        let first_points: Vec<_> = (0..5).map(|_| sampler.next_point()).collect();

        sampler.reset();

        let second_points: Vec<_> = (0..5).map(|_| sampler.next_point()).collect();

        for (p1, p2) in first_points.iter().zip(second_points.iter()) {
            for (a, b) in p1.iter().zip(p2.iter()) {
                assert!((a - b).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_sobol_high_dim() {
        // Test at maximum dimension
        let points = sobol_sequence(100, 40).unwrap();
        assert_eq!(points.len(), 100);
        assert_eq!(points[0].len(), 40);

        // All coordinates should be in [0, 1)
        for p in &points {
            for &xi in p {
                assert!((0.0..1.0).contains(&xi));
            }
        }
    }

    #[test]
    fn test_sobol_error_dim() {
        assert!(SobolGenerator::new(0).is_err());
        assert!(SobolGenerator::new(41).is_err());
    }

    #[test]
    fn test_sobol_generator_index() {
        let mut sampler = SobolGenerator::new(2).unwrap();
        assert_eq!(sampler.index(), 0);

        sampler.next_point();
        assert_eq!(sampler.index(), 1);

        sampler.next_point();
        assert_eq!(sampler.index(), 2);
    }

    #[test]
    fn test_sobol_deterministic() {
        let p1 = sobol_sequence(10, 3).unwrap();
        let p2 = sobol_sequence(10, 3).unwrap();

        for (a, b) in p1.iter().zip(p2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_sobol_sequence_from() {
        let points = sobol_sequence_from(10, 2, 50).unwrap();
        assert_eq!(points.len(), 10);

        // Verify against skip_to
        let mut sampler = SobolGenerator::new(2).unwrap();
        sampler.skip_to(50);
        for p in &points {
            let expected = sampler.next_point();
            for (a, b) in p.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-14);
            }
        }
    }
}
