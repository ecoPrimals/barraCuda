// SPDX-License-Identifier: AGPL-3.0-only

//! Banded Smith-Waterman local alignment with affine gap penalties (f64).
//!
//! Implements the classic Smith & Waterman (1981) local alignment algorithm
//! with banded DP (O(n·w) space) and affine gap penalties (Gotoh 1982).
//!
//! ## GPU Strategy
//!
//! Anti-diagonal wavefront: cells with the same `row + col` value are
//! independent given prior diagonals.  The shader processes one
//! anti-diagonal per dispatch; the Rust wrapper sweeps `d = 2..=n+m`.
//!
//! For large sequences (n, m > 512), submit via a
//! [`wgpu::CommandEncoder`] loop to amortise per-submission overhead.
//!
//! ## Absorbed from
//!
//! wetSpring handoff §Shader Design 1 (Feb 2026) — used by `bio::sate_alignment`
//! for phylogenetic multiple sequence alignment (Liu 2009 `SATé`).

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

// ─── GPU parameter struct (matches WGSL SwParams layout) ─────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SwParamsGpu {
    n: u32,
    m: u32,
    band_width: u32,
    diagonal: u32,
    gap_open: f64,
    gap_extend: f64,
}

// ─── Public alignment configuration ──────────────────────────────────────────

/// Affine gap penalty parameters for Smith-Waterman alignment.
#[derive(Debug, Clone)]
pub struct SwConfig {
    /// Penalty for opening a gap (subtracted once when a gap starts).
    pub gap_open: f64,
    /// Penalty for extending a gap (subtracted for each additional gap position).
    pub gap_extend: f64,
    /// Maximum distance from the main diagonal to consider.
    /// `0` = full DP (no banding, O(n·m) space).
    pub band_width: u32,
}

impl Default for SwConfig {
    fn default() -> Self {
        Self {
            gap_open: 11.0,
            gap_extend: 1.0,
            band_width: 64,
        }
    }
}

/// Result of a Smith-Waterman alignment.
#[derive(Debug, Clone)]
pub struct SwResult {
    /// Best local alignment score.
    pub score: f64,
    /// Row index (query position, 1-based) of the best alignment endpoint.
    pub row: usize,
    /// Column index (target position, 1-based) of the best alignment endpoint.
    pub col: usize,
}

// ─── Main operator ────────────────────────────────────────────────────────────

/// GPU-accelerated banded Smith-Waterman local alignment (f64).
///
/// # Example
///
/// ```rust,ignore
/// # use barracuda::prelude::WgpuDevice;
/// # use barracuda::ops::bio::smith_waterman::{SmithWatermanGpu, SwConfig};
/// # crate::device::test_pool::tokio_block_on(async {
/// let device = WgpuDevice::new().await.unwrap();
/// // DNA sequences encoded as 0=A, 1=C, 2=G, 3=T
/// let query  = vec![0u32, 1, 2, 3];   // ACGT
/// let target = vec![0u32, 1, 2, 3];   // ACGT
/// // BLOSUM-like substitution matrix (4×4, DNA)
/// let subst = vec![
///      2.0_f64, -1.0, -1.0, -1.0,
///     -1.0,  2.0, -1.0, -1.0,
///     -1.0, -1.0,  2.0, -1.0,
///     -1.0, -1.0, -1.0,  2.0,
/// ];
/// let sw = SmithWatermanGpu::new(&device);
/// let result = sw.align(&query, &target, &subst, &SwConfig::default()).unwrap();
/// println!("score = {}", result.score);
/// # });
/// ```
pub struct SmithWatermanGpu {
    device: Arc<WgpuDevice>,
}

impl SmithWatermanGpu {
    /// Create Smith-Waterman aligner.
    #[must_use]
    pub fn new(device: &WgpuDevice) -> Self {
        Self {
            device: Arc::new(device.clone()),
        }
    }

    /// Run banded Smith-Waterman alignment on the GPU.
    ///
    /// # Arguments
    /// - `query`:  nucleotide (or amino-acid) indices `0..alphabet_size`, length n
    /// - `target`: nucleotide (or amino-acid) indices, length m
    /// - `subst`:  substitution score matrix, flat row-major [alphabet × alphabet]
    /// - `config`: gap penalties and band width
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn align(
        &self,
        query: &[u32],
        target: &[u32],
        subst: &[f64],
        config: &SwConfig,
    ) -> Result<SwResult> {
        let dev = &self.device;
        let n = query.len();
        let m = target.len();
        let rows_cols = (n + 1) * (m + 1);

        // ── Upload buffers ─────────────────────────────────────────────────────
        let query_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SW query"),
                contents: bytemuck::cast_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let target_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SW target"),
                contents: bytemuck::cast_slice(target),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let subst_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SW subst"),
                contents: bytemuck::cast_slice(subst),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let zeros: Vec<f64> = vec![0.0; rows_cols];
        let h_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SW H"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let e_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SW E"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let f_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SW F"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE,
            });

        const SHADER: &str = include_str!("../../shaders/bio/smith_waterman_banded_f64.wgsl");

        // ── Sweep anti-diagonals d = 2 .. n+m (inclusive) ─────────────────────
        for d in 2u32..=(n + m) as u32 {
            // Number of cells on this diagonal within [1..n] × [1..m]
            let i_lo = d.saturating_sub(m as u32).max(1);
            let i_hi = (d - 1).min(n as u32);
            if i_hi < i_lo {
                continue;
            }
            let cells = i_hi - i_lo + 1;

            let params_data = SwParamsGpu {
                n: n as u32,
                m: m as u32,
                band_width: config.band_width,
                diagonal: d,
                gap_open: config.gap_open,
                gap_extend: config.gap_extend,
            };
            // Params use storage read (not uniform) because they contain f64 fields.
            let params_buf = dev
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let wg = cells.div_ceil(WORKGROUP_SIZE_1D);
            ComputeDispatch::new(dev, "SW BandedF64")
                .shader(SHADER, "main")
                .f64()
                .storage_read(0, &params_buf)
                .storage_read(1, &query_buf)
                .storage_read(2, &target_buf)
                .storage_read(3, &subst_buf)
                .storage_rw(4, &h_buf)
                .storage_rw(5, &e_buf)
                .storage_rw(6, &f_buf)
                .dispatch(wg, 1, 1)
                .submit()?;
        }

        // ── Read back H matrix and find best score on CPU ─────────────────────
        let h_data = dev.read_buffer_f64(&h_buf, rows_cols)?;

        let mut best_score = 0.0_f64;
        let mut best_row = 0usize;
        let mut best_col = 0usize;
        for i in 1..=n {
            for j in 1..=m {
                let v = h_data[i * (m + 1) + j];
                if v > best_score {
                    best_score = v;
                    best_row = i;
                    best_col = j;
                }
            }
        }

        Ok(SwResult {
            score: best_score,
            row: best_row,
            col: best_col,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    fn dna_subst() -> Vec<f64> {
        // Simple +2 match / -1 mismatch for ACGT
        let mut m = vec![-1.0_f64; 16];
        m[0] = 2.0;
        m[5] = 2.0;
        m[10] = 2.0;
        m[15] = 2.0;
        m
    }

    #[tokio::test]
    async fn test_identical_sequences() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let sw = SmithWatermanGpu::new(&device);
        // ACGT vs ACGT: perfect match, score = 4 × 2.0 = 8.0
        let q = vec![0u32, 1, 2, 3];
        let t = vec![0u32, 1, 2, 3];
        let result = sw
            .align(
                &q,
                &t,
                &dna_subst(),
                &SwConfig {
                    gap_open: 11.0,
                    gap_extend: 1.0,
                    band_width: 0,
                },
            )
            .unwrap();
        assert!((result.score - 8.0).abs() < 0.01, "score={}", result.score);
        assert_eq!(result.row, 4);
        assert_eq!(result.col, 4);
    }

    #[tokio::test]
    async fn test_single_base_match() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let sw = SmithWatermanGpu::new(&device);
        // Only A matches
        let q = vec![0u32, 3, 3, 3]; // ATTT
        let t = vec![3u32, 3, 0, 3]; // TTAT
        let result = sw
            .align(&q, &t, &dna_subst(), &SwConfig::default())
            .unwrap();
        // Best match: A(q[0]) vs A(t[2]) or T runs → at least score 2.0
        assert!(result.score >= 2.0, "score={}", result.score);
    }

    #[tokio::test]
    async fn test_no_match() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let sw = SmithWatermanGpu::new(&device);
        // All mismatches → score 0 (local alignment floor)
        let q = vec![0u32, 0, 0]; // AAA
        let t = vec![3u32, 3, 3]; // TTT
        let result = sw
            .align(&q, &t, &dna_subst(), &SwConfig::default())
            .unwrap();
        assert_eq!(result.score, 0.0);
    }
}
