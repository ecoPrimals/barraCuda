// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spectral theory for discrete Schrödinger operators.
//!
//! Implements lattice Hamiltonians and spectral analysis tools:
//!
//! - **`CsrMatrix` + `SpMV`**: sparse matrix-vector product (GPU primitive via existing `spmv_f64.wgsl`)
//! - **Lanczos eigensolve**: Krylov tridiagonalization with full reorthogonalization
//! - **Anderson model**: random potential in 1D, 2D, and 3D
//!   - 1D/2D: all states localized (Abrahams et al. 1979)
//!   - 3D: genuine metal-insulator transition with mobility edge (`W_c` ≈ 16.5)
//! - **Almost-Mathieu operator**: quasiperiodic potential, Aubry-André transition
//! - **Transfer matrix**: Lyapunov exponent computation
//! - **Tridiagonal eigensolve**: Sturm bisection for all eigenvalues
//! - **Level statistics**: spacing ratio for localization diagnostics
//!
//! # Physics
//!
//! The 1D discrete Schrödinger equation on ℤ:
//!   ψ_{n+1} + ψ_{n-1} + `V_n` `ψ_n` = E `ψ_n`
//!
//! is equivalent to the eigenvalue problem for the tridiagonal matrix
//! H with diagonal `V_i` and off-diagonal −1. The spectral properties of H
//! (eigenvalues, eigenvectors, Lyapunov exponent) determine transport:
//! extended states → metallic, localized states → insulating.
//!
//! # Provenance
//!
//! - Anderson (1958) "Absence of diffusion in certain random lattices"
//! - Aubry & André (1980) "Analyticity breaking and Anderson localization"
//! - Jitomirskaya (1999) "Metal-insulator transition for the almost Mathieu operator"
//! - Herman (1983) "Une méthode pour minorer les exposants de Lyapunov"
//! - Avila (2015) "Global theory of one-frequency Schrödinger operators" (Fields Medal)
//! - Kappus & Wegner (1981) "Anomaly in the band centre of the 1D Anderson model"
//!
//! Absorbed from hotSpring v0.6.0 (Kachkovskiy spectral theory extension)

pub mod anderson;
#[cfg(feature = "gpu")]
pub mod batch_ipr;
mod hofstadter;
pub(crate) mod lanczos;
pub(crate) mod sparse;
mod stats;
mod tridiag;
#[cfg(feature = "gpu")]
pub mod tridiag_eigh_gpu;

pub use anderson::{
    AndersonSweepPoint, anderson_2d, anderson_3d, anderson_3d_correlated, anderson_4d,
    anderson_eigenvalues, anderson_hamiltonian, anderson_potential, anderson_sweep_averaged,
    clean_2d_lattice, clean_3d_lattice, find_w_c, lyapunov_averaged, lyapunov_exponent,
    wegner_block_4d,
};
#[cfg(feature = "gpu")]
pub use batch_ipr::BatchIprGpu;
pub use hofstadter::{GOLDEN_RATIO, almost_mathieu_hamiltonian, gcd, hofstadter_butterfly};
pub use lanczos::{LanczosTridiag, lanczos, lanczos_eigenvalues};
pub use sparse::{SpectralCsrMatrix, WGSL_SPMV_CSR_F64};
pub use stats::{
    GOE_R, POISSON_R, SpectralAnalysis, SpectralPhase, classify_spectral_phase, detect_bands,
    level_spacing_ratio, spectral_bandwidth, spectral_condition_number,
};
#[cfg(feature = "gpu")]
pub use stats::{WeightMatrixAnalysis, analyze_weight_matrix};
pub use tridiag::{find_all_eigenvalues, sturm_count, tridiag_eigenvectors};
#[cfg(feature = "gpu")]
pub use tridiag_eigh_gpu::BatchedTridiagEighGpu;
