// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cache-aligned N-dimensional tile decomposition for GPU lattice workloads.
//!
//! Tile sizes derive from measured on-chip cache capacity in [`SiliconProfile`]:
//! AMD Infinity Cache (~128 MB → ~125 MB tiles, e.g. 16⁴ lattice sites) vs
//! NVIDIA L2 (~6 MB → ~5.5 MB tiles, e.g. 6⁴ lattice sites).

use super::silicon_profile::{GpuVendorTag, SiliconProfile};
use serde::{Deserialize, Serialize};

/// Fraction of measured cache capacity used per tile (headroom for tags/prefetch).
const CACHE_USABLE_NUMERATOR: u64 = 125;
const CACHE_USABLE_DENOMINATOR: u64 = 128;

/// Default dimensionality when inferring cubic/hypercubic tile shapes.
const DEFAULT_TILE_RANK: u32 = 4;

/// Decomposes N-dimensional grids into cache-aligned tiles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileDecomposer {
    tile_shape: Vec<u32>,
    halo_width: u32,
}

/// One tile in a decomposed grid.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileSpec {
    /// Origin offset in each grid dimension.
    pub offset: Vec<u32>,
    /// Extent in each dimension (may be smaller than `tile_shape` at boundaries).
    pub shape: Vec<u32>,
    /// Halo ghost-cell width for stencil exchange.
    pub halo_width: u32,
}

impl TileDecomposer {
    /// Create a decomposer for a given profile's cache hierarchy.
    ///
    /// Tiles are sized to fit the GPU's largest on-chip cache (`l2_bytes` or
    /// `infinity_cache_bytes`, whichever is larger).
    #[must_use]
    pub fn from_profile(profile: &SiliconProfile, element_bytes: u32) -> Self {
        Self::from_profile_with_rank(profile, element_bytes, DEFAULT_TILE_RANK)
    }

    /// Create a decomposer with explicit grid rank (number of dimensions).
    #[must_use]
    pub fn from_profile_with_rank(profile: &SiliconProfile, element_bytes: u32, rank: u32) -> Self {
        let cache_bytes = profile.l2_bytes.max(profile.infinity_cache_bytes);
        let tile_shape = compute_tile_shape(cache_bytes, element_bytes, rank);
        Self {
            tile_shape,
            halo_width: u32::from(profile.vendor != GpuVendorTag::Software),
        }
    }

    /// Reference tile shape used for decomposition.
    #[must_use]
    pub fn tile_shape(&self) -> &[u32] {
        &self.tile_shape
    }

    /// Decompose a grid into tiles, returning `(offset, shape)` per tile.
    #[must_use]
    pub fn decompose(&self, grid_shape: &[u32]) -> Vec<TileSpec> {
        if grid_shape.is_empty() {
            return Vec::new();
        }

        let rank = grid_shape.len();
        let mut tile_dims = self.tile_shape.clone();
        if tile_dims.len() < rank {
            let fill = *tile_dims.last().unwrap_or(&1);
            tile_dims.resize(rank, fill);
        } else if tile_dims.len() > rank {
            tile_dims.truncate(rank);
        }

        let mut offsets = vec![0u32; rank];
        let mut tiles = Vec::new();
        decompose_recursive(
            grid_shape,
            &tile_dims,
            &mut offsets,
            0,
            &mut tiles,
            self.halo_width,
        );
        tiles
    }

    /// Number of tiles for a given grid.
    #[must_use]
    pub fn tile_count(&self, grid_shape: &[u32]) -> usize {
        if grid_shape.is_empty() {
            return 0;
        }

        grid_shape
            .iter()
            .zip(self.tile_shape.iter().chain(std::iter::repeat(&1)))
            .map(|(grid, tile)| tile_count_axis(*grid, *tile))
            .product()
    }
}

#[must_use]
fn compute_tile_shape(cache_bytes: u64, element_bytes: u32, rank: u32) -> Vec<u32> {
    let rank = rank.max(1);
    if element_bytes == 0 || cache_bytes == 0 {
        return vec![1; rank as usize];
    }

    let usable_cache =
        cache_bytes.saturating_mul(CACHE_USABLE_NUMERATOR) / CACHE_USABLE_DENOMINATOR;
    let elements_per_tile = usable_cache / u64::from(element_bytes);
    if elements_per_tile == 0 {
        return vec![1; rank as usize];
    }

    let side = nth_root(elements_per_tile, rank).max(1);
    vec![side; rank as usize]
}

#[must_use]
fn nth_root(value: u64, n: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    let n = n.max(1);
    let mut lo = 1u64;
    let mut hi = value;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if pow_u64(mid, n) <= value {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo.min(u64::from(u32::MAX)) as u32
}

#[must_use]
fn pow_u64(base: u64, exp: u32) -> u64 {
    let mut result = 1u64;
    for _ in 0..exp {
        result = result.saturating_mul(base);
    }
    result
}

#[must_use]
fn tile_count_axis(grid: u32, tile: u32) -> usize {
    if tile == 0 {
        return 0;
    }
    grid.div_ceil(tile) as usize
}

fn decompose_recursive(
    grid_shape: &[u32],
    tile_dims: &[u32],
    offset: &mut [u32],
    dim: usize,
    out: &mut Vec<TileSpec>,
    halo_width: u32,
) {
    if dim == grid_shape.len() {
        let shape: Vec<u32> = grid_shape
            .iter()
            .zip(offset.iter())
            .zip(tile_dims.iter())
            .map(|((grid, off), tile)| {
                let remaining = grid.saturating_sub(*off);
                remaining.min(*tile)
            })
            .collect();
        out.push(TileSpec {
            offset: offset.to_vec(),
            shape,
            halo_width,
        });
        return;
    }

    let grid = grid_shape[dim];
    let tile = tile_dims.get(dim).copied().unwrap_or(1).max(1);
    let mut start = 0u32;
    while start < grid {
        offset[dim] = start;
        decompose_recursive(grid_shape, tile_dims, offset, dim + 1, out, halo_width);
        start = start.saturating_add(tile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn profile_with_cache(
        vendor: GpuVendorTag,
        l2_bytes: u64,
        infinity_cache_bytes: u64,
    ) -> SiliconProfile {
        SiliconProfile {
            adapter_name: "test".into(),
            vendor,
            vram_bytes: 0,
            boost_ghz: 0.0,
            units: BTreeMap::new(),
            compositions: vec![],
            df64_tflops: 0.0,
            l2_bytes,
            infinity_cache_bytes,
            tmu_count: 0,
            rop_count: 0,
            subgroup_size: 0,
            dispatch_overhead_us: 0.0,
            streaming_speedup: 1.0,
            max_1d_workgroups: 65535,
            measured_at: String::new(),
        }
    }

    #[test]
    fn nvidia_like_tile_side_six_for_reference_lattice() {
        // 6^4 sites × ~4737 B/site ≈ 5.5 MB (RTX 3090 L2 budget).
        let profile = profile_with_cache(GpuVendorTag::Nvidia, 6 * 1024 * 1024, 0);
        let decomposer = TileDecomposer::from_profile(&profile, 4737);
        assert_eq!(decomposer.tile_shape(), &[6, 6, 6, 6]);
    }

    #[test]
    fn amd_like_tile_side_sixteen_for_reference_lattice() {
        // 16^4 sites × ~2000 B/site ≈ 125 MB (6950 XT Infinity Cache budget).
        let profile = profile_with_cache(GpuVendorTag::Amd, 4 * 1024 * 1024, 128 * 1024 * 1024);
        let decomposer = TileDecomposer::from_profile(&profile, 2000);
        assert_eq!(decomposer.tile_shape(), &[16, 16, 16, 16]);
    }

    #[test]
    fn zero_cache_yields_unit_tiles() {
        let profile = profile_with_cache(GpuVendorTag::Software, 0, 0);
        let decomposer = TileDecomposer::from_profile(&profile, 8);
        assert_eq!(decomposer.tile_shape(), &[1, 1, 1, 1]);
    }

    #[test]
    fn empty_grid_returns_no_tiles() {
        let profile = profile_with_cache(GpuVendorTag::Nvidia, 6 * 1024 * 1024, 0);
        let decomposer = TileDecomposer::from_profile(&profile, 8);
        assert!(decomposer.decompose(&[]).is_empty());
        assert_eq!(decomposer.tile_count(&[]), 0);
    }

    #[test]
    fn decompose_covers_full_grid_with_boundary_clips() {
        // Small cache → 5×5 tiles in 2D so a 10×10 grid splits into four tiles.
        let profile = profile_with_cache(GpuVendorTag::Nvidia, 256, 0);
        let decomposer = TileDecomposer::from_profile_with_rank(&profile, 8, 2);
        let grid = [10u32, 10];
        let tiles = decomposer.decompose(&grid);
        assert_eq!(tiles.len(), decomposer.tile_count(&grid));
        assert_eq!(tiles.len(), 4);

        let total_cells: u64 = tiles
            .iter()
            .map(|t| t.shape.iter().map(|&s| u64::from(s)).product::<u64>())
            .sum();
        assert_eq!(total_cells, 100);
    }

    #[test]
    fn tile_count_matches_decompose_len() {
        let profile = profile_with_cache(GpuVendorTag::Amd, 0, 128 * 1024 * 1024);
        let decomposer = TileDecomposer::from_profile_with_rank(&profile, 4, 3);
        let grid = [32u32, 32, 32];
        assert_eq!(
            decomposer.decompose(&grid).len(),
            decomposer.tile_count(&grid)
        );
    }
}
