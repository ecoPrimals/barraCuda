// SPDX-License-Identifier: AGPL-3.0-only
//! Capability-based discovery for barraCuda.
//!
//! Derives capabilities, provides, and methods from the actual registered
//! IPC methods — no hardcoded values. Per wateringHole principle: the
//! discovery file is self-describing based on what the primal supports.

use crate::ipc::methods::REGISTERED_METHODS;
use std::collections::BTreeSet;

/// Extract the domain (second component) from a method name.
///
/// Method format: `barracuda.{domain}.{operation}`.
fn domain_of(method: &str) -> Option<&str> {
    method.split('.').nth(1)
}

/// Derive capability domains from registered method namespaces.
///
/// Maps method domains to capability identifiers:
/// - `compute`, `device`, `validate` → `gpu_compute`
/// - `tensor` → `tensor_ops`
/// - `fhe` → `fhe`
/// - Meta domains (`primal`, `health`, `tolerances`) → `core`
fn derive_capabilities(methods: &[&str]) -> Vec<String> {
    let mut caps = BTreeSet::new();
    for method in methods {
        if let Some(domain) = domain_of(method) {
            let cap = match domain {
                "compute" | "device" | "validate" => "gpu_compute",
                "tensor" => "tensor_ops",
                "fhe" => "fhe",
                "primal" | "health" | "tolerances" => "core",
                _ => domain,
            };
            caps.insert(cap.to_string());
        }
    }
    caps.into_iter().collect()
}

/// Derive the "provides" list from method namespaces.
///
/// Maps method domains/operations to capability IDs:
/// - Any `compute.*` method → `gpu.compute`
/// - Exact `barracuda.compute.dispatch` → `gpu.dispatch`
/// - Any `tensor.*` method → `tensor.ops`
fn derive_provides(methods: &[&str]) -> Vec<String> {
    let mut provides = BTreeSet::new();
    for method in methods {
        if let Some(domain) = domain_of(method) {
            match domain {
                "compute" => {
                    provides.insert("gpu.compute".to_string());
                    if *method == "barracuda.compute.dispatch" {
                        provides.insert("gpu.dispatch".to_string());
                    }
                }
                "tensor" => {
                    provides.insert("tensor.ops".to_string());
                }
                "device" => {
                    provides.insert("gpu.device".to_string());
                }
                _ => {}
            }
        }
    }
    provides.into_iter().collect()
}

/// Return the list of registered methods for discovery.
#[must_use]
pub fn registered_methods() -> &'static [&'static str] {
    REGISTERED_METHODS
}

/// Derive capabilities from the dispatch table.
#[must_use]
pub fn capabilities() -> Vec<String> {
    derive_capabilities(REGISTERED_METHODS)
}

/// Derive provides from the dispatch table.
#[must_use]
pub fn provides() -> Vec<String> {
    derive_provides(REGISTERED_METHODS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_capabilities() {
        let caps = capabilities();
        assert!(caps.contains(&"gpu_compute".to_string()));
        assert!(caps.contains(&"tensor_ops".to_string()));
        assert!(caps.contains(&"fhe".to_string()));
    }

    #[test]
    fn test_derive_provides() {
        let p = provides();
        assert!(p.contains(&"gpu.compute".to_string()));
        assert!(p.contains(&"tensor.ops".to_string()));
        assert!(p.contains(&"gpu.dispatch".to_string()));
    }

    #[test]
    fn test_registered_methods_non_empty() {
        let methods = registered_methods();
        assert!(!methods.is_empty());
        assert!(methods.contains(&"barracuda.compute.dispatch"));
    }
}
