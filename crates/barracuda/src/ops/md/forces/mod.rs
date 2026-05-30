// SPDX-License-Identifier: AGPL-3.0-or-later
//! Force Kernels Module
//!
//! Implementations of fundamental force calculations for molecular dynamics:
//! - Coulomb (electrostatic, f32 and f64)
//! - Yukawa (screened electrostatic, f32 and f64)
//! - Lennard-Jones (van der Waals, f32 and f64)
//! - Morse (anharmonic bonded, f32 and f64)
//! - Harmonic Bond (covalent bonds, f64)
//! - Harmonic Angle (three-body valence angle, f64)
//! - Periodic Dihedral (four-body torsion, f64)
//! - Improper Dihedral (planarity restraint, f64)
//! - Born-Mayer (hard-core repulsion)
//!
//! ## WGSL as Unified Math Language (Feb 16, 2026)
//!
//! All force kernels are pure WGSL → WebGPU → Vulkan:
//! - Same code runs on NVIDIA, AMD, Intel
//! - Native f64 builtins (sqrt, exp) at full hardware speed
//! - No CUDA lock-in, no `OpenCL` fragmentation
//!
//! ## f64 Evolution
//!
//! - `yukawa_f64.wgsl`: All-pairs O(N²) with PBC minimum-image ✅
//! - `yukawa_celllist_f64.wgsl`: 27-neighbor cell-list O(N) ✅
//! - `lennard_jones_f64.wgsl`: Van der Waals with shifted potential ✅
//! - `coulomb_f64.wgsl`: Electrostatics with Ewald real-space ✅
//! - `morse_f64.wgsl`: Bonded interactions with energy ✅
//! - `harmonic_bond_f64.wgsl`: Standard covalent bond stretching ✅
//! - `harmonic_angle_f64.wgsl`: Three-body valence angle bending ✅
//! - `dihedral_f64.wgsl`: Periodic proper dihedral torsion ✅
//! - `improper_f64.wgsl`: Harmonic improper dihedral (planarity) ✅
//!
//! **Deep Debt**: All math in WGSL, zero unsafe

pub mod born_mayer;
pub mod born_mayer_f64;
pub mod coulomb;
pub mod coulomb_f64;
pub mod dihedral_f64;
pub mod harmonic_angle_f64;
pub mod harmonic_bond_f64;
pub mod improper_f64;
pub mod lennard_jones;
pub mod lennard_jones_f64;
pub mod morse;
pub mod morse_f64;
pub mod yukawa;
pub mod yukawa_celllist_f64;
pub mod yukawa_f64;

pub use born_mayer::BornMayerForce;
pub use born_mayer_f64::BornMayerForceF64;
pub use coulomb::CoulombForce;
pub use coulomb_f64::CoulombForceF64;
pub use dihedral_f64::{DihedralTorsion, DihedralTorsionF64};
pub use harmonic_angle_f64::{HarmonicAngle, HarmonicAngleF64};
pub use harmonic_bond_f64::{HarmonicBond, HarmonicBondF64};
pub use improper_f64::{ImproperDihedral, ImproperDihedralF64};
pub use lennard_jones::LennardJonesForce;
pub use lennard_jones_f64::LennardJonesF64;
pub use morse::MorseForce;
pub use morse_f64::MorseForceF64;
pub use yukawa::YukawaForce;
pub use yukawa_celllist_f64::YukawaCellListF64;
pub use yukawa_f64::YukawaForceF64;
