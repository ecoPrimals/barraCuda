// SPDX-License-Identifier: AGPL-3.0-or-later
//! Property-Based Tests - Mathematical correctness validation
//!
//! **Purpose**: Verify fundamental mathematical properties hold across inputs
//! **Coverage**: FHE operations, gradients, numerical stability
//! **Deep Debt**: Correctness guarantees, no silent bugs

pub mod fhe_properties;
pub mod fhe_proptest;