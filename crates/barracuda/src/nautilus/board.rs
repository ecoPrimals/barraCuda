// SPDX-License-Identifier: AGPL-3.0-only
//! Board structure and response for `BingoCube` reservoir computing.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// Configuration for a `BingoCube` board.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardConfig {
    /// Grid size (L×L).
    pub grid_size: usize,
    /// Range per column: column k draws from [k*range, (k+1)*range).
    pub range_per_column: u32,
    /// Seed for deterministic generation.
    pub seed: u64,
}

impl Default for BoardConfig {
    fn default() -> Self {
        Self {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        }
    }
}

/// Reservoir input: discrete (Bingo-style) or continuous (hash-projected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReservoirInput {
    /// Discrete Bingo-style input (cell values that match).
    Discrete(Vec<u32>),
    /// Continuous features (hash-projected to activations).
    Continuous(Vec<f64>),
}

/// Response vector from a board (L² activations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseVector {
    /// Activation values (length L²).
    pub activations: Vec<f64>,
}

/// `BingoCube` board: L×L grid with column-range constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Board {
    /// Cell values [row][col]; each column has unique values in its range.
    pub cells: Vec<Vec<u32>>,
    /// Board configuration.
    pub config: BoardConfig,
}

impl Board {
    /// Create a deterministic board from seed.
    #[must_use]
    pub fn from_seed(config: BoardConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let l = config.grid_size;
        let range = config.range_per_column;
        let mut cells = vec![vec![0u32; l]; l];

        for k in 0..l {
            let lo = range * (k as u32);
            let hi = range * ((k + 1) as u32);
            let mut col_vals: Vec<u32> = (lo..hi).collect();
            col_vals.shuffle(&mut rng);
            for row in 0..l {
                cells[row][k] = col_vals[row];
            }
        }

        Self {
            cells,
            config: BoardConfig {
                grid_size: config.grid_size,
                range_per_column: config.range_per_column,
                seed: config.seed,
            },
        }
    }

    /// Compute response to reservoir input.
    #[must_use]
    pub fn respond(&self, input: &ReservoirInput) -> ResponseVector {
        let l = self.config.grid_size;
        let mut activations = vec![0.0; l * l];

        match input {
            ReservoirInput::Discrete(vals) => {
                let center = l / 2;
                for i in 0..l {
                    for j in 0..l {
                        let cell = self.cells[i][j];
                        let idx = i * l + j;
                        if i == center && j == center {
                            activations[idx] = 0.5;
                        } else if vals.contains(&cell) {
                            activations[idx] = 1.0;
                        } else {
                            activations[idx] = 0.0;
                        }
                    }
                }
            }
            ReservoirInput::Continuous(features) => {
                for i in 0..l {
                    for j in 0..l {
                        let cell = self.cells[i][j];
                        let mut hasher = blake3::Hasher::new();
                        hasher.update(b"NAUTILUS_PROJ");
                        hasher.update(&(i as u32).to_le_bytes());
                        hasher.update(&(j as u32).to_le_bytes());
                        hasher.update(&cell.to_le_bytes());
                        for f in features {
                            hasher.update(&f.to_le_bytes());
                        }
                        let hash = hasher.finalize();
                        let b = hash.as_bytes();
                        let u =
                            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                        activations[i * l + j] = (u as f64) / (u64::MAX as f64);
                    }
                }
            }
        }

        ResponseVector { activations }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_deterministic() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 42,
        };
        let b1 = Board::from_seed(config.clone(), 12345);
        let b2 = Board::from_seed(config, 12345);
        assert_eq!(b1.cells, b2.cells);
    }

    #[test]
    fn test_board_column_range() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let board = Board::from_seed(config, 999);
        let range = 15u32;
        for k in 0..board.config.grid_size {
            let lo = range * (k as u32);
            let hi = range * ((k + 1) as u32);
            for row in 0..board.config.grid_size {
                let cell = board.cells[row][k];
                assert!(
                    cell >= lo && cell < hi,
                    "cell at ({row},{k}) = {cell} not in [{lo}, {hi})"
                );
            }
        }
    }

    #[test]
    fn test_board_no_duplicates_per_column() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 7,
        };
        let board = Board::from_seed(config, 111);
        for k in 0..board.config.grid_size {
            let mut seen = std::collections::HashSet::new();
            for row in 0..board.config.grid_size {
                let cell = board.cells[row][k];
                assert!(seen.insert(cell), "duplicate {cell} in column {k}");
            }
        }
    }

    #[test]
    fn test_discrete_response() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let board = Board::from_seed(config, 42);
        let center = board.config.grid_size / 2;
        let l = board.config.grid_size;

        // Use values that exist in the board
        let cell_at_center = board.cells[center][center];
        let other_cell = if center > 0 {
            board.cells[center - 1][0]
        } else {
            board.cells[1][0]
        };
        let input = ReservoirInput::Discrete(vec![cell_at_center, other_cell]);

        let r = board.respond(&input);
        assert_eq!(r.activations.len(), l * l);

        for i in 0..l {
            for j in 0..l {
                let idx = i * l + j;
                let val = r.activations[idx];
                if i == center && j == center {
                    assert!((val - 0.5).abs() < 1e-10, "center should be 0.5");
                } else if val > 0.5 {
                    assert!((val - 1.0).abs() < 1e-10, "match should be 1.0");
                } else {
                    assert!((val - 0.0).abs() < 1e-10, "no match should be 0.0");
                }
            }
        }
    }

    #[test]
    fn test_continuous_response() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let board = Board::from_seed(config, 77);
        let input = ReservoirInput::Continuous(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let r = board.respond(&input);
        assert_eq!(
            r.activations.len(),
            board.config.grid_size * board.config.grid_size
        );

        for &v in &r.activations {
            assert!(
                (0.0..=1.0).contains(&v),
                "continuous response should be in [0,1], got {v}"
            );
        }
    }

    #[test]
    fn test_different_boards_different_responses() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let b1 = Board::from_seed(config.clone(), 100);
        let b2 = Board::from_seed(config, 200);

        let input = ReservoirInput::Continuous(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let r1 = b1.respond(&input);
        let r2 = b2.respond(&input);

        assert_ne!(r1.activations, r2.activations);
    }
}
