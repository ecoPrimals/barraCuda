// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fault Injection Tests - Error scenarios and graceful degradation
//!
//! **Purpose**: Validate error handling under failure conditions
//! **Coverage**: Invalid inputs, boundary cases, error propagation
//! **Deep Debt**: Graceful errors (Result), no panics

pub mod invalid_inputs;
pub mod boundary_cases;
pub mod error_propagation;
pub mod fhe_fault_tests;
pub mod fhe_binary_ops_tests;
pub mod fhe_logical_ops_tests;