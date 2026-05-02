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
pub(crate) fn provided_capability_groups(version: &str) -> Vec<serde_json::Value> {
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

/// Meta/housekeeping domains excluded from discovery-service capability registration.
///
/// These are ecosystem probe or infrastructure domains, not functional
/// capabilities that other primals would resolve via `ipc.resolve`.
const DISCOVERY_EXCLUDED_DOMAINS: &[&str] = &[
    "health",
    "capabilities",
    "identity",
    "primal",
    "tolerances",
    "validate",
    "compute",
];

/// Derive semantic capability domain tags for `ipc.register` with the
/// discovery service (`DISCOVERY_SOCKET`).
///
/// Returns the unique first-segment domains from registered methods, excluding
/// meta/housekeeping domains (health, capabilities, identity, etc.).
/// Per Phase 55b: these are the functional capability tags that other primals
/// resolve via `ipc.resolve`.
#[must_use]
pub fn discovery_capability_domains() -> Vec<String> {
    let mut domains = BTreeSet::new();
    for method in REGISTERED_METHODS {
        if let Some(domain) = domain_of(method) {
            if !DISCOVERY_EXCLUDED_DOMAINS.contains(&domain) {
                domains.insert(domain.to_string());
            }
        }
    }
    domains.into_iter().collect()
}

/// Self-register capabilities with the discovery service via `DISCOVERY_SOCKET`.
///
/// Per Phase 55b: primals self-register at startup so the discovery service can
/// resolve capabilities for other primals via `ipc.resolve`. Fire-and-forget —
/// any failure is logged at debug level and does not block startup.
#[cfg(unix)]
pub async fn register_with_discovery(endpoint: &str) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let Ok(socket_var) = std::env::var("DISCOVERY_SOCKET") else {
        return;
    };
    let socket_path = std::path::Path::new(&socket_var);
    if !socket_path.exists() {
        tracing::debug!(path = %socket_path.display(), "DISCOVERY_SOCKET absent — skip register");
        return;
    }

    let capabilities = discovery_capability_domains();
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "ipc.register",
        "params": {
            "primal_id": crate::PRIMAL_NAMESPACE,
            "capabilities": capabilities,
            "endpoint": endpoint,
        },
        "id": 1
    });

    let Ok(stream) = tokio::net::UnixStream::connect(socket_path).await else {
        tracing::debug!("discovery service connect failed — skipping registration");
        return;
    };
    let mut reader = BufReader::new(stream);
    let Ok(mut line) = serde_json::to_string(&request) else {
        return;
    };
    line.push('\n');
    if reader.get_mut().write_all(line.as_bytes()).await.is_err()
        || reader.get_mut().flush().await.is_err()
    {
        tracing::debug!("discovery service write failed — registration may not have arrived");
        return;
    }

    let mut response_line = String::new();
    if reader.read_line(&mut response_line).await.is_err() {
        return;
    }
    if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&response_line) {
        if let Some(vep) = resp
            .get("result")
            .and_then(|r| r.get("virtual_endpoint"))
            .and_then(|v| v.as_str())
        {
            tracing::info!(virtual_endpoint = vep, domains = ?capabilities, "registered via DISCOVERY_SOCKET");
        } else if let Some(err) = resp.get("error") {
            tracing::debug!(?err, "discovery service registration returned error");
        }
    }
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
        "stats" => "Statistical operations (mean, std_dev, variance, correlation, weighted_mean)",
        "noise" => "Noise generation (Perlin 2D/3D)",
        "rng" => "Random number generation",
        "tensor" => "GPU tensor operations (create, matmul, add, scale, clamp, reduce, sigmoid)",
        "fhe" => "Fully homomorphic encryption primitives (NTT, pointwise_mul)",
        "linalg" => "Linear algebra (solve, eigenvalues — CPU inline-data path)",
        "spectral" => "Spectral analysis (FFT, power spectrum — CPU inline-data path)",
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
    fn discovery_capabilities_include_functional_domains() {
        let caps = discovery_capability_domains();
        for expected in &[
            "tensor",
            "math",
            "stats",
            "linalg",
            "ml",
            "spectral",
            "activation",
            "noise",
            "rng",
            "fhe",
            "device",
        ] {
            assert!(
                caps.contains(&(*expected).to_string()),
                "discovery capabilities should include {expected}"
            );
        }
    }

    #[test]
    fn discovery_capabilities_exclude_meta_domains() {
        let caps = discovery_capability_domains();
        for excluded in DISCOVERY_EXCLUDED_DOMAINS {
            assert!(
                !caps.contains(&(*excluded).to_string()),
                "discovery capabilities should NOT include meta domain {excluded}"
            );
        }
    }

    #[test]
    fn discovery_capabilities_derived_not_hardcoded() {
        let caps = discovery_capability_domains();
        let mut expected_domains = BTreeSet::new();
        for method in REGISTERED_METHODS {
            if let Some(domain) = domain_of(method) {
                if !DISCOVERY_EXCLUDED_DOMAINS.contains(&domain) {
                    expected_domains.insert(domain.to_string());
                }
            }
        }
        let expected: Vec<String> = expected_domains.into_iter().collect();
        assert_eq!(caps, expected, "discovery caps must match derived domains");
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
