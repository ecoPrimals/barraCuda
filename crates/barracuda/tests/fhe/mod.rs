// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unit Tests for FHE WGSL Shaders
//!
//! **Philosophy**:
//! - Test each shader in isolation
//! - Cover all code paths (>80% coverage target)
//! - Test edge cases and boundaries
//! - Fast execution (<1s per test)
//! - Property-based testing for mathematical guarantees
//!
//! **Deep Debt Compliance**:
//! - Pure Rust (no unsafe)
//! - Clear error messages
//! - Deterministic tests
//! - No flaky tests

mod error_handling;
mod fast_poly_mul;
mod helpers;
mod intt;
mod ntt;
mod performance;
mod pointwise;
