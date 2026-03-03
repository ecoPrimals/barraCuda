// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision Tests - FP32 accuracy validation
//!
//! **Purpose**: Validate numerical correctness against CPU reference
//! **Coverage**: Compare GPU vs CPU, check FP32 precision bounds
//! **Deep Debt**: Reproducible results, no silent accuracy loss

pub mod core_ops;
pub mod activations;
pub mod convolutions;
