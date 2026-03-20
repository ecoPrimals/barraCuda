// SPDX-License-Identifier: AGPL-3.0-or-later

//! Population pharmacokinetics GPU primitives.
//!
//! Absorbed from healthSpring V14 (Mar 2026). Provides GPU-accelerated
//! FOCE gradient computation, VPC Monte Carlo simulation, and related
//! population PK operations.
//!
//! | Module | Shader | Primitive |
//! |--------|--------|-----------|
//! | `foce_gradient` | `foce_gradient_f64.wgsl` | Per-subject FOCE gradient (f64) |
//! | `vpc_simulate` | `vpc_simulate_f64.wgsl` | VPC Monte Carlo PK simulation (f64) |

pub mod foce_gradient;
pub mod vpc_simulate;
