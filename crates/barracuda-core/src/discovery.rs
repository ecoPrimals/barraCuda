// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based discovery for barraCuda.
//!
//! Derives capabilities, provides, and methods from the actual registered
//! IPC methods — no hardcoded values. Per wateringHole principle: the
//! discovery file is self-describing based on what the primal supports.
//! Method names follow the wateringHole semantic standard (`{domain}.{operation}`)
//! — the primal only has self-knowledge and discovers others at runtime.

use crate::ipc::methods::REGISTERED_METHODS;
use std::collections::BTreeSet;

/// Extract the domain (first component) from a semantic method name.
///
/// Method format: `{domain}.{operation}`.
fn domain_of(method: &str) -> Option<&str> {
    method.split('.').next()
}

/// Derive capability domains from registered method namespaces.
///
/// Maps method domains to capability identifiers:
/// - `compute`, `device`, `validate` -> `gpu_compute`
/// - `tensor` -> `tensor_ops`
/// - `fhe` -> `fhe`
/// - Meta domains (`primal`, `health`, `tolerances`) -> `core`
fn derive_capabilities(methods: &[&str]) -> Vec<String> {
    let mut caps = BTreeSet::new();
    for method in methods {
        if let Some(domain) = domain_of(method) {
            let cap = match domain {
                "compute" | "device" | "validate" => "gpu_compute",
                "tensor" => "tensor_ops",
                "fhe" => "fhe",
                "primal" | "health" | "tolerances" => "core",
                other => other,
            };
            caps.insert(cap.to_string());
        }
    }
    caps.into_iter().collect()
}

/// Derive the "provides" list from method namespaces.
///
/// Maps method domains/operations to capability IDs:
/// - Any `compute.*` method -> `gpu.compute`
/// - `compute.dispatch` -> `gpu.dispatch`
/// - Any `tensor.*` method -> `tensor.ops`
fn derive_provides(methods: &[&str]) -> Vec<String> {
    let mut provides = BTreeSet::new();
    for method in methods {
        if let Some(domain) = domain_of(method) {
            match domain {
                "compute" => {
                    provides.insert("gpu.compute".to_string());
                    if *method == "compute.dispatch" {
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
pub fn registered_methods() -> Vec<String> {
    REGISTERED_METHODS
        .iter()
        .map(|s| (*s).to_string())
        .collect()
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

/// Wire Standard L2/L3: structured capability groups for `provided_capabilities`.
///
/// Groups registered methods by domain, emitting `{type, methods, version,
/// description}` objects. Descriptions derive from the domain name — no
/// hardcoded domain catalog.
#[must_use]
pub fn provided_capability_groups(version: &str) -> Vec<serde_json::Value> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for method in REGISTERED_METHODS {
        if let Some(dot) = method.find('.') {
            let domain = &method[..dot];
            let operation = &method[dot + 1..];
            groups.entry(domain).or_default().push(operation);
        }
    }
    groups
        .into_iter()
        .map(|(domain, methods)| {
            serde_json::json!({
                "type": domain,
                "methods": methods,
                "version": version,
                "description": domain_description(domain),
            })
        })
        .collect()
}

fn domain_description(domain: &str) -> &'static str {
    match domain {
        "health" => "Ecosystem health probes (liveness, readiness, check)",
        "capabilities" => "Capability self-advertisement",
        "primal" => "Primal identity and meta-capabilities",
        "device" => "GPU device enumeration and probing",
        "tolerances" => "Numerical tolerance configuration",
        "validate" => "GPU stack validation",
        "compute" => "GPU compute shader dispatch",
        "math" => "CPU math and activation functions",
        "activation" => "Cognitive activation functions (Fitts, Hick)",
        "stats" => "Statistical operations (mean, std_dev, weighted_mean)",
        "noise" => "Noise generation (Perlin 2D/3D)",
        "rng" => "Random number generation",
        "tensor" => "GPU tensor operations (create, matmul, add, scale, clamp, reduce, sigmoid)",
        "fhe" => "Fully homomorphic encryption primitives (NTT, pointwise_mul)",
        "identity" => "Primal identity for observability",
        _ => "Capability domain",
    }
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
        assert!(caps.contains(&"core".to_string()));
    }

    #[test]
    fn test_capabilities_includes_all_domains() {
        let caps = capabilities();
        assert!(
            caps.contains(&"capabilities".to_string()),
            "capabilities domain should be derived from capabilities.list method"
        );
    }

    #[test]
    fn test_derive_provides() {
        let p = provides();
        assert!(p.contains(&"gpu.compute".to_string()));
        assert!(p.contains(&"tensor.ops".to_string()));
        assert!(p.contains(&"gpu.dispatch".to_string()));
        assert!(p.contains(&"gpu.device".to_string()));
    }

    #[test]
    fn test_registered_methods_semantic_format() {
        let methods = registered_methods();
        assert!(!methods.is_empty());
        for m in &methods {
            assert!(
                m.contains('.'),
                "method {m} should use domain.operation format"
            );
            assert!(
                !m.starts_with(&format!("{}.", crate::PRIMAL_NAMESPACE)),
                "method {m} should NOT have primal namespace prefix on the wire"
            );
        }
    }

    #[test]
    fn test_registered_methods_contains_dispatch() {
        let methods = registered_methods();
        assert!(methods.contains(&"compute.dispatch".to_string()));
    }

    #[test]
    fn test_registered_methods_contains_ecosystem_probes() {
        let methods = registered_methods();
        assert!(methods.contains(&"health.liveness".to_string()));
        assert!(methods.contains(&"health.readiness".to_string()));
        assert!(methods.contains(&"health.check".to_string()));
        assert!(methods.contains(&"capabilities.list".to_string()));
    }

    #[test]
    fn test_domain_of_helper() {
        assert_eq!(domain_of("health.check"), Some("health"));
        assert_eq!(domain_of("compute.dispatch"), Some("compute"));
        assert_eq!(domain_of("nodot"), Some("nodot"));
        assert_eq!(domain_of(""), Some(""));
    }

    #[test]
    fn test_derive_capabilities_custom_methods() {
        let caps = derive_capabilities(&["custom.thing", "compute.dispatch"]);
        assert!(caps.contains(&"custom".to_string()));
        assert!(caps.contains(&"gpu_compute".to_string()));
    }

    #[test]
    fn test_derive_provides_no_compute() {
        let provides = derive_provides(&["health.check", "primal.info"]);
        assert!(provides.is_empty());
    }

    #[test]
    fn test_derive_provides_device_domain() {
        let provides = derive_provides(&["device.list"]);
        assert!(provides.contains(&"gpu.device".to_string()));
    }

    #[test]
    fn test_provided_capability_groups_structure() {
        let groups = provided_capability_groups("0.3.11");
        assert!(!groups.is_empty());
        for group in &groups {
            assert!(group["type"].is_string(), "group must have type");
            assert!(group["methods"].is_array(), "group must have methods");
            assert_eq!(group["version"], "0.3.11");
            assert!(
                group["description"].is_string(),
                "group must have description"
            );
        }
    }

    #[test]
    fn test_provided_capability_groups_contains_compute() {
        let groups = provided_capability_groups("0.3.11");
        let domains: Vec<&str> = groups.iter().filter_map(|g| g["type"].as_str()).collect();
        assert!(domains.contains(&"compute"));
        assert!(domains.contains(&"tensor"));
        assert!(domains.contains(&"fhe"));
        assert!(domains.contains(&"health"));
    }
}
