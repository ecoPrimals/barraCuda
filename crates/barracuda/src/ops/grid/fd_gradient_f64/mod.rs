// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finite-Difference Gradient GPU Implementation (f64)
//!
//! GPU-accelerated gradient and Laplacian operations on structured grids.
//! Uses WGSL shaders for f64 precision on all GPU hardware.
//!
//! **Smart Refactoring**: Uses `fd_common` for shared infrastructure.

mod cylindrical;
mod gradient_1d;
mod gradient_2d;
mod laplacian_2d;

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests;

pub use cylindrical::{CylindricalGradient, CylindricalLaplacian};
pub use gradient_1d::Gradient1D;
pub use gradient_2d::Gradient2D;
pub use laplacian_2d::Laplacian2D;
