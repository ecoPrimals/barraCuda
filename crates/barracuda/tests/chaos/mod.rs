// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chaos Tests - Random inputs, stress, concurrent execution
//!
//! **Purpose**: Find edge case bugs through randomization
//! **Coverage**: Random dimensions, concurrent ops, stress tests
//! **Deep Debt**: No hardcoded assumptions, discover failures
//!
//! ## Test Categories
//!
//! 1. **random_inputs** - Fuzz testing with random data
//! 2. **stress** - Heavy load and resource pressure
//! 3. **concurrent** - Race conditions and deadlocks
//! 4. **device_chaos** - Device failures, VRAM exhaustion, fallbacks
//! 5. **fhe_chaos** - FHE-specific chaos scenarios

pub mod concurrent;
pub mod device_chaos;
pub mod fhe_chaos_expanded;
pub mod fhe_chaos_tests;
pub mod random_inputs;
pub mod stress;