// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage-focused tests for JSON-RPC method handlers.
//!
//! These tests exercise validation error paths that are reachable because
//! handlers validate input parameters *before* checking device availability.
//! This makes every validation branch testable without GPU hardware.
//!
//! Organised by domain following the Sprint 37 `methods_tests/` pattern.

mod compute_tests;
mod fhe_tests;
mod math_stats_tests;
mod noise_rng_activation_tests;
mod tensor_tests;
mod type_validation_tests;

use crate::BarraCudaPrimal;

pub(super) fn test_primal() -> BarraCudaPrimal {
    BarraCudaPrimal::new()
}
