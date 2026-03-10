// SPDX-License-Identifier: AGPL-3.0-only
//! Time Integrators Module
//!
//! Implementations of numerical integration schemes for ODEs/PDEs:
//! - Velocity-Verlet (symplectic, MD, f32)
//! - Split Velocity-Verlet (kick-drift-kick, f64) — hotSpring validated
//! - RK4 (high accuracy, general ODEs)
//! - Laplacian stencil (PDEs, mesh operations)
//!
//! **hotSpring Integration** (Feb 2026):
//! - f64 split VV for flexible thermostating
//! - Explicit PBC wrap during drift
//! - Standard in LAMMPS/GROMACS
//!
//! **Deep Debt**: All math in WGSL, zero unsafe

pub mod laplacian;
pub mod rk4;
pub mod velocity_verlet;
pub mod velocity_verlet_f64;
pub mod velocity_verlet_split_f64;

pub use laplacian::Laplacian;
pub use rk4::Rk4;
pub use velocity_verlet::VelocityVerlet;
pub use velocity_verlet_f64::VelocityVerletF64;
pub use velocity_verlet_split_f64::{VelocityVerletHalfKick, VelocityVerletKickDrift};
