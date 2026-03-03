//! Thermostats for Molecular Dynamics
//!
//! Temperature control for equilibration and NVT ensembles.
//!
//! **Available Thermostats**:
//! - Berendsen: Weak coupling, equilibration only (does NOT sample canonical)
//! - Nosé-Hoover: Deterministic NVT, properly samples canonical ensemble
//! - Langevin: Stochastic, friction + noise, good for non-equilibrium/Brownian
//!
//! **Usage Pattern**:
//! 1. Equilibration: Use Berendsen with τ ≈ 5*dt for fast relaxation
//! 2. Production NVT (deterministic): Nosé-Hoover with τ ≈ 100*dt
//! 3. Production NVT (stochastic): Langevin with γ ≈ 1/τ
//! 4. Production NVE: Remove thermostat entirely
//!
//! **Comparison**:
//! | Thermostat | Canonical? | Deterministic? | Best For |
//! |------------|------------|----------------|----------|
//! | Berendsen | No | Yes | Fast equilibration |
//! | Nosé-Hoover | Yes | Yes | Production NVT |
//! | Langevin | Yes | No | Brownian/implicit solvent |
//!
//! **Deep Debt Compliance**:
//! - ✅ WGSL shader-first (separate .wgsl files)
//! - ✅ Full f64 precision
//! - ✅ Zero unsafe code

mod berendsen;
mod langevin;
mod nose_hoover;

pub use berendsen::BerendsenThermostat;
pub use langevin::{LangevinParams, LangevinStep};
pub use nose_hoover::{NoseHooverChain, NoseHooverHalfKick};
