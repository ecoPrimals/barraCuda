// SPDX-License-Identifier: AGPL-3.0-or-later
//! Selection, crossover, and mutation for evolutionary reservoir computing.

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::board::Board;

/// Selection method for evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMethod {
    Elitism(usize),
    Tournament(usize),
    RouletteWheel,
}

/// Evolution configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub selection: SelectionMethod,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            selection: SelectionMethod::Elitism(4),
            mutation_rate: 0.15,
            crossover_rate: 0.7,
        }
    }
}

/// Evolve a new population from parents using fitness.
pub fn evolve(
    parents: &[Board],
    fitness: &[f64],
    config: &EvolutionConfig,
    pop_size: usize,
    rng: &mut impl Rng,
) -> Vec<Board> {
    let mut next = Vec::with_capacity(pop_size);

    // Selection
    let selected = match &config.selection {
        SelectionMethod::Elitism(k) => {
            let mut idx_fit: Vec<(usize, f64)> =
                fitness.iter().enumerate().map(|(i, &f)| (i, f)).collect();
            idx_fit.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx_fit
                .iter()
                .take(*k)
                .map(|(i, _)| parents[*i].clone())
                .collect()
        }
        SelectionMethod::Tournament(k) => {
            let mut sel = Vec::new();
            while sel.len() < pop_size {
                let best: usize = (0..*k)
                    .map(|_| rng.gen_range(0..parents.len()))
                    .max_by(|&a, &b| {
                        fitness[a]
                            .partial_cmp(&fitness[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_default();
                sel.push(parents[best].clone());
            }
            sel
        }
        SelectionMethod::RouletteWheel => {
            let sum: f64 = fitness.iter().map(|f| f.max(0.0) + 1e-10).sum();
            let mut sel = Vec::new();
            for _ in 0..pop_size {
                let r: f64 = rng.gen();
                let mut acc = 0.0;
                let mut pushed = false;
                for (i, &f) in fitness.iter().enumerate() {
                    acc += (f.max(0.0) + 1e-10) / sum;
                    if r <= acc {
                        sel.push(parents[i].clone());
                        pushed = true;
                        break;
                    }
                }
                if !pushed {
                    sel.push(parents[parents.len() - 1].clone());
                }
            }
            sel
        }
    };

    // Elitism: keep top individuals first
    let n_elite = match &config.selection {
        SelectionMethod::Elitism(k) => *k,
        _ => 0,
    };
    for i in 0..n_elite.min(pop_size) {
        if i < selected.len() {
            next.push(selected[i].clone());
        }
    }

    // Crossover + mutation for rest
    while next.len() < pop_size {
        let a_idx = rng.gen_range(0..parents.len());
        let b_idx = rng.gen_range(0..parents.len());
        let child = if rng.gen::<f64>() < config.crossover_rate && a_idx != b_idx {
            crossover_columns(&parents[a_idx], &parents[b_idx], rng)
        } else {
            parents[a_idx].clone()
        };
        let mut child = child;
        mutate(&mut child, config.mutation_rate, rng);
        next.push(child);
    }

    next.truncate(pop_size);
    next
}

/// Crossover by swapping random columns between parents.
pub fn crossover_columns(a: &Board, b: &Board, rng: &mut impl Rng) -> Board {
    let l = a.config.grid_size;
    let mut cells = a.cells.clone();
    for k in 0..l {
        if rng.gen::<bool>() {
            for row in 0..l {
                cells[row][k] = b.cells[row][k];
            }
        }
    }
    Board {
        cells,
        config: a.config.clone(),
    }
}

/// Mutate board cells within column range (preserves no-duplicates per column).
pub fn mutate(board: &mut Board, rate: f64, rng: &mut impl Rng) {
    let l = board.config.grid_size;
    let range = board.config.range_per_column;
    for k in 0..l {
        let col_lo = range * (k as u32);
        let col_hi = range * ((k + 1) as u32);
        for row in 0..l {
            if rng.gen::<f64>() < rate {
                let used: std::collections::HashSet<u32> =
                    (0..l).map(|r| board.cells[r][k]).collect();
                let available: Vec<u32> = (col_lo..col_hi).filter(|v| !used.contains(v)).collect();
                if !available.is_empty() {
                    let new_val = available[rng.gen_range(0..available.len())];
                    board.cells[row][k] = new_val;
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::nautilus::board::BoardConfig;

    #[test]
    fn test_elitism_preserves_best() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let parents: Vec<Board> = (0..8)
            .map(|i| Board::from_seed(config.clone(), 100 + i))
            .collect();
        let fitness = vec![0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6];
        let evo_config = EvolutionConfig {
            selection: SelectionMethod::Elitism(2),
            mutation_rate: 0.0,
            crossover_rate: 0.0,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let next = evolve(&parents, &fitness, &evo_config, 8, &mut rng);
        assert_eq!(next.len(), 8);
        assert_eq!(next[0].cells, parents[1].cells);
        assert_eq!(next[1].cells, parents[5].cells);
    }

    #[test]
    fn test_mutation_preserves_column_range() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let mut board = Board::from_seed(config.clone(), 555);
        let mut rng = StdRng::seed_from_u64(123);
        mutate(&mut board, 0.5, &mut rng);

        let range = config.range_per_column;
        for k in 0..board.config.grid_size {
            let lo = range * (k as u32);
            let hi = range * ((k + 1) as u32);
            for row in 0..board.config.grid_size {
                let cell = board.cells[row][k];
                assert!(
                    cell >= lo && cell < hi,
                    "mutated cell at ({row},{k}) = {cell} not in [{lo}, {hi})"
                );
            }
        }
    }

    #[test]
    fn test_crossover_preserves_column_range() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let a = Board::from_seed(config.clone(), 111);
        let b = Board::from_seed(config.clone(), 222);
        let mut rng = StdRng::seed_from_u64(456);
        let child = crossover_columns(&a, &b, &mut rng);

        let range = config.range_per_column;
        for k in 0..child.config.grid_size {
            let lo = range * (k as u32);
            let hi = range * ((k + 1) as u32);
            for row in 0..child.config.grid_size {
                let cell = child.cells[row][k];
                assert!(
                    cell >= lo && cell < hi,
                    "crossover cell at ({row},{k}) = {cell} not in [{lo}, {hi})"
                );
            }
        }
    }

    #[test]
    fn test_population_size_preserved() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let parents: Vec<Board> = (0..10)
            .map(|i| Board::from_seed(config.clone(), 200 + i))
            .collect();
        let fitness: Vec<f64> = (0..10).map(|i| (i as f64) * 0.1).collect();
        let evo_config = EvolutionConfig::default();
        let mut rng = StdRng::seed_from_u64(789);
        let next = evolve(&parents, &fitness, &evo_config, 10, &mut rng);
        assert_eq!(next.len(), 10);
    }
}
