//! Property-based and mathematical correctness tests for BarraCuda operations.
//!
//! `fhe_proptest`  — proptest-driven randomised NTT/modular arithmetic invariants
//!                   (pure, no GPU required).
//!
//! `fhe_properties` — GPU-side FHE integration tests validating mathematical
//!                    properties (NTT linearity, homomorphic ops, key switch,
//!                    modulus switch, rotation composition). Tests skip
//!                    gracefully when no GPU is present.

#[path = "property/fhe_proptest.rs"]
mod fhe_proptest;

#[path = "property/fhe_properties.rs"]
mod fhe_properties;
