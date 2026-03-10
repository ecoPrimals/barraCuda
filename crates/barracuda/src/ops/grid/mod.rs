// SPDX-License-Identifier: AGPL-3.0-only
//! Structured Grid Operations
//!
//! **Physics-Agnostic Grid Primitives**
//!
//! Finite-difference and interpolation operations on structured grids.
//!
//! ## Operations
//!
//! - **Gradient**: First derivatives via central/forward/backward FD
//! - **Laplacian**: Second derivatives (∇²)
//! - **Gradient magnitude**: |∇f|
//! - **Coordinate systems**: Cartesian 1D/2D/3D, Cylindrical (ρ,z)
//!
//! ## Applications
//!
//! - **Fluid dynamics**: Navier-Stokes, pressure Poisson
//! - **Heat transfer**: Diffusion equation
//! - **Electrostatics**: Poisson, Laplace equations
//! - **Wave propagation**: Acoustic, seismic, EM
//! - **Image processing**: Edge detection, smoothing
//! - **Nuclear physics**: HFB density gradients (validated by hotSpring)
//!
//! ## Stencil Accuracy
//!
//! - **Interior**: 2nd order central difference (3-point stencil)
//! - **Boundaries**: 1st order forward/backward (2-point stencil)
//!
//! Higher-order stencils (4th, 6th order) can be added as needed.
//!
//! ## Grid Conventions
//!
//! - **Row-major**: index = ix * ny + iy (C-style)
//! - **Cylindrical**: ρ starts at dρ (not 0), z centered
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::ops::grid::{Gradient2D, Laplacian2D};
//!
//! let grad = Gradient2D::new(device, nx, ny, dx, dy)?;
//! let (gx, gy) = grad.compute(&field).await?;
//!
//! let laplacian = Laplacian2D::new(device, nx, ny, dx, dy)?;
//! let nabla2_f = laplacian.compute(&field).await?;
//! ```
//!
//! ## Deep Debt Compliance
//!
//! - Pure WGSL (f64 via `SHADER_F64`)
//! - Physics-agnostic (no domain-specific parameters)
//! - Validated by hotSpring nuclear EOS (169/169 acceptance checks)

mod agro_ops;
mod fd_common;
mod fd_gradient_f64;
mod grid_search_ops;
mod spin_orbit_f64;

pub use agro_ops::{batched_crop_pipeline, dual_kc, hargreaves_et0, van_genuchten};
pub use fd_common::{
    FdComputeRunner, FdPipelineBuilder, create_empty_f64_buffer, create_f64_buffer,
    create_staging_buffer,
};
pub use fd_gradient_f64::{
    CylindricalGradient, CylindricalLaplacian, Gradient1D, Gradient2D, Laplacian2D,
};
pub use grid_search_ops::{GridSearchResult, band_edges_parallel, grid_fit_2d, grid_search_3d};
pub use spin_orbit_f64::{SpinOrbitGpu, compute_ls_factor};
