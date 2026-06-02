// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nuclear structure GPU primitives (HFB, BCS pairing, Skyrme potentials).
//!
//! Absorbed from hotSpring v0.6.4 handoff (Feb 2026); shader consolidation
//! completed Jun 2026.
//!
//! | Module | Content |
//! |--------|---------|
//! | `hfb` | Spherical HFB: BCS bisection, density, energy, Hamiltonian, potentials |
//! | `hfb_deformed` | Axially-deformed HFB: BCS, density, energy, Hamiltonian, potentials, wavefunctions |
//! | `semf` | SEMF binding energy + chi-squared batch shaders |
//! | `plasma` | BGK relaxation, Euler HLL, Mermin dielectric shaders |
//! | `fes` | Metadynamics free-energy surface Gaussian summation |
//! | `absorbed_shaders` | Re-exports all 19 physics WGSL shader constants |

pub mod absorbed_shaders;
pub mod fes;
pub mod hfb;
pub mod hfb_deformed;
pub mod plasma;
pub mod semf;

pub use absorbed_shaders::*;
