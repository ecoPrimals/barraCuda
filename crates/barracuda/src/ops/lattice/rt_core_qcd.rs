// SPDX-License-Identifier: AGPL-3.0-or-later

//! RT core QCD mappings — capability-first exploration targets.
//!
//! Hardware RT cores are BVH-accelerated ray-geometry intersection engines.
//! They perform spatial queries in O(log n) time via dedicated silicon that
//! runs concurrently with shader ALU. This module documents the theorized
//! and benchmarked mappings to lattice QCD operations.
//!
//! # Hardware Capability
//!
//! RT cores accelerate two operations:
//! 1. **BVH build** — construct a bounding volume hierarchy over geometry
//! 2. **BVH query** — find ray-geometry intersections (traversal + leaf test)
//!
//! On regular lattices, neighbor lookup is O(1) (modular index arithmetic),
//! so RT cores provide no benefit for standard nearest-neighbor stencils.
//! However, they become relevant when the geometry is:
//! - **Irregular** (deformed lattices, improved actions with extended stencils)
//! - **Adaptive** (multigrid hierarchies, AMR)
//! - **High-dimensional** (parameter space, not physical space)
//!
//! # Benchmarked Performance
//!
//! From `bench_rt_core_probe.rs` (hotSpring, August 2026):
//! - RTX 3090: 1.5 Mtriangles/s BVH build, sub-µs per ray query
//! - RX 6950 XT: 0.1 Mtriangles/s (software BVH, no hardware RT)
//!
//! # QCD Mapping 1: Wilson Loop Path Tracing
//!
//! A Wilson loop IS a path through gauge links. On a regular lattice the path
//! is trivially enumerable, but on deformed/large lattices (L > 64), long
//! Wilson loops (perimeter >> L) intersect O(L) links. The RT core can
//! accelerate "which links does this path intersect?" via:
//!
//! ```text
//! 1. Build BVH: each link midpoint → AABB leaf (one per 4×volume links)
//! 2. For each Wilson loop segment: cast ray → BVH returns intersected link IDs
//! 3. Multiply ordered product of U_link along path
//! ```
//!
//! **When this wins**: Deformed lattices where link positions are non-uniform
//! (e.g. anisotropic improved actions, gravitational backgrounds). On a regular
//! 32⁴ lattice, O(1) index arithmetic is faster than BVH traversal overhead.
//!
//! **Exploration target**: Prototype with wgpu ray-query extension on RTX 3090.
//! Measure breakeven lattice size (expected L > 64 or non-cubic geometries).
//!
//! # QCD Mapping 2: Parameter-Space Nearest-Neighbor (Hot Start BVH)
//!
//! Before thermalizing a new configuration at (β, m_q, seed), query a BVH
//! built over previously cached thermalized configurations to find the nearest
//! neighbor in parameter space. Use that config as a hot start → dramatically
//! reduced thermalization time.
//!
//! ```text
//! 1. Build BVH: each cached config → point in (β, m_q, κ, ...) space
//! 2. New campaign: cast ray from target parameters → nearest cached config
//! 3. Load nearest config, resume HMC from there (skip 80-500 warmup trajectories)
//! ```
//!
//! **When this wins**: Large ensembles with many parameter points (e.g. finite-T
//! scans across 50+ β values). The BVH makes nearest-config lookup O(log N)
//! instead of O(N) linear scan over the config archive.
//!
//! **Exploration target**: Build BVH over strandGate's existing cached configs
//! (currently 75/87 in `hotspring.thermalization`). Measure time saved vs
//! cold-start thermalization.
//!
//! # QCD Mapping 3: Multigrid Coarsening Queries
//!
//! On adaptive multigrid lattices, the fine-to-coarse mapping ("which coarse
//! cell contains this fine-grid point?") requires a spatial query. With
//! precomputed tables this is O(1), but table construction itself is O(N).
//! RT cores can answer this query without precomputation:
//!
//! ```text
//! 1. Build BVH over coarse-cell AABBs
//! 2. For each fine point: ray-query → which coarse cell contains it
//! 3. Prolongation/restriction operators use this mapping
//! ```
//!
//! **When this wins**: Dynamic multigrid where the coarse grid changes every
//! few solver iterations (adaptive MG for near-critical slowing-down). Static
//! multigrid uses precomputed tables (O(1) lookup) which is faster.
//!
//! **Exploration target**: Prototype with `gpu_cg_resident` multigrid
//! preconditioner. Measure BVH rebuild cost vs precomputed table rebuild for
//! adaptive coarsening schedules.
//!
//! # Deployment Status
//!
//! **Not deployed**: Regular 32⁴ lattice uses O(1) index arithmetic for all
//! neighbor lookups. RT cores sit idle during current pure-gauge campaigns.
//!
//! **Path to deployment**:
//! 1. wgpu ray-query extension stabilizes (currently experimental)
//! 2. Campaign workload evolves to deformed geometries or large parameter scans
//! 3. Multigrid preconditioner goes adaptive (dynamical fermion campaign)
//!
//! # References
//!
//! - `springs/hotSpring/barracuda/src/bin/bench_rt_core_probe.rs` — BVH benchmarks
//! - `infra/whitePaper/subGen/GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md` — theory
//! - `infra/whitePaper/subGen/SILICON_FOLD_DEEP_EXPLORATION_AUG09_2026.md` — census

/// RT core exploration status for telemetry/reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtCoreStatus {
    /// Hardware supports ray-query (NVIDIA SM75+)
    Available,
    /// Software fallback only (AMD RDNA2 without RT hardware, or RDNA2 with
    /// limited ray-accelerator that doesn't expose wgpu ray-query)
    SoftwareFallback,
    /// No RT capability detected
    Unavailable,
}

/// Categorized QCD use cases for RT cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtQcdMapping {
    /// Wilson loop path tracing on deformed/large lattices.
    WilsonLoopTracing,
    /// Parameter-space BVH for hot-start config selection.
    ParameterSpaceBvh,
    /// Multigrid coarsening cell lookup on adaptive grids.
    MultigridCoarsening,
}

impl RtQcdMapping {
    /// Whether this mapping is competitive on regular periodic lattices.
    #[inline]
    #[must_use]
    pub const fn competitive_on_regular_lattice(&self) -> bool {
        false
    }

    /// Minimum lattice extent where RT acceleration is expected to break even
    /// with O(1) index arithmetic (estimated from BVH traversal overhead).
    #[inline]
    #[must_use]
    pub const fn estimated_breakeven_l(&self) -> u32 {
        match self {
            Self::WilsonLoopTracing => 64,
            Self::ParameterSpaceBvh => 1,
            Self::MultigridCoarsening => 32,
        }
    }

    /// Human-readable description of the mapping.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::WilsonLoopTracing => {
                "Cast rays along Wilson loop paths; BVH returns intersected link IDs on deformed geometries"
            }
            Self::ParameterSpaceBvh => {
                "BVH over (beta, mass, seed) space; nearest-neighbor query for hot-start config selection"
            }
            Self::MultigridCoarsening => {
                "BVH over coarse-cell AABBs; fine-point containment query for adaptive multigrid"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mappings_not_competitive_on_regular() {
        assert!(!RtQcdMapping::WilsonLoopTracing.competitive_on_regular_lattice());
        assert!(!RtQcdMapping::ParameterSpaceBvh.competitive_on_regular_lattice());
        assert!(!RtQcdMapping::MultigridCoarsening.competitive_on_regular_lattice());
    }

    #[test]
    fn breakeven_estimates() {
        assert_eq!(RtQcdMapping::WilsonLoopTracing.estimated_breakeven_l(), 64);
        assert_eq!(RtQcdMapping::ParameterSpaceBvh.estimated_breakeven_l(), 1);
        assert_eq!(RtQcdMapping::MultigridCoarsening.estimated_breakeven_l(), 32);
    }

    #[test]
    fn descriptions_non_empty() {
        assert!(!RtQcdMapping::WilsonLoopTracing.description().is_empty());
        assert!(!RtQcdMapping::ParameterSpaceBvh.description().is_empty());
        assert!(!RtQcdMapping::MultigridCoarsening.description().is_empty());
    }
}
