// SPDX-License-Identifier: AGPL-3.0-only

//! Nuclear structure GPU primitives (HFB, BCS pairing, Skyrme potentials).
//!
//! Absorbed from hotSpring v0.6.4 handoff (Feb 2026).
//!
//! | Module | Content |
//! |--------|---------|
//! | `hfb` | Spherical HFB: BCS bisection, density, energy, Hamiltonian, potentials |
//! | `hfb_deformed` | Axially-deformed HFB: BCS, density, energy, Hamiltonian, potentials, wavefunctions |

pub mod hfb;
pub mod hfb_deformed;
