//! Molecular Dynamics Operations
//!
//! Operations for molecular dynamics simulations:
//! - Periodic boundary conditions (PBC)
//! - Force kernels (Coulomb, Yukawa, LJ, etc.)
//! - Time integrators (Velocity-Verlet, RK4, split VV)
//! - Thermostats (Berendsen, Nosé-Hoover, Langevin)
//! - Neighbor search (cell list for O(N) scaling)
//! - Observables (kinetic energy, RDF, VACF, SSF, MSD)
//! - Electrostatics (PPPM/Ewald for long-range Coulomb)
//!
//! **hotSpring Integration** (Feb 2026):
//! - f64 Yukawa force with PBC + PE (9/9 Sarkas cases validated)
//! - Split Velocity-Verlet (kick-drift-kick pattern)
//! - Berendsen thermostat for equilibration
//! - Nosé-Hoover thermostat for NVT production
//! - Langevin thermostat for stochastic dynamics
//! - Cell list for O(N) neighbor search
//! - Per-particle kinetic energy for temperature
//! - MSD for diffusion coefficient calculation
//!
//! **PPPM/Ewald** (In Progress):
//! - Parameter auto-tuning for target accuracy
//! - Architecture defined, awaiting FFT f64 evolution
//!
//! **Deep Debt Compliance**: All math in WGSL, zero unsafe

pub mod electrostatics;
pub mod forces;
pub mod integrators;
pub mod neighbor;
pub mod observables;
pub mod pbc;
pub mod stress_virial;
pub mod thermostats;
pub mod vacf;

pub use electrostatics::{PppmAccuracy, PppmParams};
pub use forces::*;
pub use integrators::*;
pub use neighbor::{CellList, CellListGpu};
pub use observables::{
    compute_msd, compute_rdf, compute_ssf, compute_vacf, KineticEnergy, KineticEnergyF64, Msd, Rdf,
    RdfHistogramF64, SsfGpu, Vacf, VacfGpu,
};
pub use pbc::{DistanceMetric, PbcDistance};
pub use stress_virial::compute_stress_virial;
pub use thermostats::{
    BerendsenThermostat, LangevinParams, LangevinStep, NoseHooverChain, NoseHooverHalfKick,
};
pub use vacf::compute_vacf_batch;

// Re-export for convenience
pub use pbc::PbcDistance as Pbc;
