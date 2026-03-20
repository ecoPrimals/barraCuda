// SPDX-License-Identifier: AGPL-3.0-or-later
//! Partial Differential Equation Solvers
//!
//! This module provides numerical methods for solving PDEs commonly
//! encountered in scientific computing and physics simulations.
//!
//! # Solvers
//!
//! - **Crank-Nicolson**: Implicit 2nd-order scheme for diffusion equations
//! - **Richards**: 1D unsaturated zone water flow (van Genuchten-Mualem)
//!
//! # Applications
//!
//! - **Two-Temperature Model (TTM)**: Ultrafast laser heating
//! - **Heat diffusion**: Thermal transport simulations
//! - **Schrödinger equation**: Time-dependent quantum mechanics
//! - **Richards equation**: Soil water infiltration (airSpring, wetSpring)
//!
//! # References
//!
//! - Numerical Recipes, §19.2
//! - J. Crank & P. Nicolson (1947), "A practical method for numerical
//!   evaluation of solutions of partial differential equations"
//! - van Genuchten (1980), "A closed-form equation for predicting the
//!   hydraulic conductivity of unsaturated soils"

pub mod crank_nicolson;
pub mod richards;
pub mod richards_gpu;

pub use crank_nicolson::{
    CrankNicolson1D, CrankNicolsonConfig, HeatEquation1D, WGSL_CRANK_NICOLSON_F64,
    crank_nicolson_step,
};
pub use richards::{RichardsBc, RichardsConfig, RichardsResult, SoilParams, solve_richards};
pub use richards_gpu::RichardsGpu;
