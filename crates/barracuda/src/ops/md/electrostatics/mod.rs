//! Long-Range Electrostatics (PPPM/Ewald)
//!
//! **Purpose**: Accurate Coulomb interactions for periodic systems
//!
//! **Algorithm**: Particle-Particle Particle-Mesh (PPPM)
//!
//! PPPM splits Coulomb interactions into two parts:
//! 1. **Short-range (PP)**: Direct Coulomb with erfc damping
//! 2. **Long-range (PM)**: FFT-based mesh solve for k > 0
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    PPPM Electrostatics                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  1. Charge Spreading (particles → mesh)                        │
//! │     └── B-spline interpolation to mesh nodes                   │
//! │                                                                 │
//! │  2. Forward FFT (mesh → k-space)                               │
//! │     └── Uses existing Fft3D infrastructure                     │
//! │                                                                 │
//! │  3. Green's Function (k-space multiplication)                  │
//! │     └── ρ̃(k) × G(k) where G(k) = 4π/k² × influence function  │
//! │                                                                 │
//! │  4. Backward FFT (k-space → mesh)                              │
//! │     └── Get mesh potential φ(r)                                │
//! │                                                                 │
//! │  5. Force Interpolation (mesh → particles)                     │
//! │     └── B-spline gradient interpolation                        │
//! │                                                                 │
//! │  6. Short-Range Correction                                     │
//! │     └── erfc-damped direct Coulomb for nearby pairs            │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Ewald Splitting
//!
//! The Coulomb potential 1/r is split using:
//! ```text
//! 1/r = erfc(αr)/r + erf(αr)/r
//!       \_________/   \________/
//!       short-range   long-range
//!       (real space)  (k-space via FFT)
//! ```
//!
//! The splitting parameter α controls the crossover:
//! - Large α → more work in k-space (fewer mesh modes needed)
//! - Small α → more work in real space (larger cutoff needed)
//!
//! Optimal α balances real-space cutoff and mesh size.
//!
//! # PPPM Parameters
//!
//! | Parameter | Symbol | Typical Value | Notes |
//! |-----------|--------|---------------|-------|
//! | Splitting | α | box_side/6 | Balances PP/PM work |
//! | Mesh size | Kx,Ky,Kz | 32-128 | Power of 2 for FFT |
//! | Interpolation order | p | 4-7 | Higher = more accurate |
//! | Real cutoff | rc | box_side/4 | Works with cell-list |
//! | Accuracy | δ | 1e-5 | Force relative error |
//!
//! # Implementation Status
//!
//! | Component | Status | Notes |
//! |-----------|--------|-------|
//! | Fft3D (f32) | ✅ Done | |
//! | Fft1D (f64) | ✅ Done | For PPPM precision |
//! | B-spline functions | ✅ Done | Cardinal B-splines with derivatives |
//! | Charge spreading | ✅ Done | CPU implementation |
//! | Green's function | ✅ Done | Precomputed with influence correction |
//! | Force interpolation | ✅ Done | B-spline gradient method |
//! | erfc short-range | ✅ Done | With self-energy and dipole corrections |
//! | Combined PPPM | ✅ Done | CPU reference impl, GPU FFT ready |
//! | **GPU PPPM** | ✅ Done | Universal WGSL shaders |
//!
//! # WGSL Shaders (Universal)
//!
//! All PPPM math is implemented in WGSL for hardware-agnostic execution:
//! - `bspline.wgsl` - B-spline evaluation kernel
//! - `charge_spread.wgsl` - Particle → mesh spreading
//! - `greens_apply.wgsl` - K-space Green's function
//! - `force_interp.wgsl` - Mesh → particle interpolation
//! - `erfc_forces.wgsl` - Real-space short-range forces
//!
//! # References
//!
//! - Essmann et al., JCP 103 (1995) - Smooth PME algorithm
//! - Deserno & Holm, JCP 109 (1998) - PPPM accuracy analysis
//! - LAMMPS PPPM implementation (BSD-3-Clause)

pub(crate) mod bspline;
pub(crate) mod charge_spread;
pub(crate) mod force_interpolation;
mod greens_function;
mod pppm;
mod pppm_buffers; // EVOLVED: Extracted for modularity (Feb 14, 2026)
mod pppm_gpu;
mod pppm_layouts; // EVOLVED: Extracted for modularity (Feb 14, 2026)
mod pppm_params;
mod short_range;

// CPU reference implementation
pub use bspline::{bspline, bspline_deriv, influence_function, BsplineCoeffs};
pub use charge_spread::{spread_charges, spread_charges_with_coeffs, ChargeMesh};
pub use force_interpolation::{
    interpolate_forces, interpolate_forces_from_positions, PotentialMesh,
};

// WGSL shader sources (for custom GPU pipelines)
pub use bspline::WGSL_BSPLINE_F64;
pub use charge_spread::WGSL_CHARGE_SPREAD_F64;
pub use force_interpolation::WGSL_FORCE_INTERPOLATION_F64;
pub use greens_function::GreensFunction;
pub use pppm::{Pppm, PppmError};
pub use pppm_params::{PppmAccuracy, PppmParams};
pub use short_range::{
    compute_short_range, compute_short_range_forces, dipole_correction, erfc, erfc_deriv,
    self_energy_correction,
};

// GPU universal implementation (WGSL shaders)
pub use pppm_gpu::PppmGpu;

// GPU helpers (for advanced users)
pub use pppm_buffers::{PppmBuffers, PppmCpuFft};
pub use pppm_layouts::{PppmBindGroupLayouts, PppmLayouts, PppmPipelines};
