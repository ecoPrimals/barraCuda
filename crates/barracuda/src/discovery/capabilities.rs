// SPDX-License-Identifier: AGPL-3.0-or-later

//! Capability extraction from diverse primal response formats.
//!
//! Normalises 6 different JSON capability shapes into `Vec<String>`.

/// Extract capabilities from any known primal response format (A–F).
///
/// Use when the payload shape is unknown (e.g. `capabilities.list` result).
#[must_use]
pub fn extract_from_any(value: &serde_json::Value) -> Vec<String> {
    if let Some(arr) = value.as_array() {
        return extract_from_array(arr);
    }

    let mut caps = Vec::new();

    if let Some(provided) = value
        .get("provided_capabilities")
        .and_then(serde_json::Value::as_array)
    {
        for entry in provided {
            if let Some(cap_type) = entry.get("type").and_then(serde_json::Value::as_str) {
                caps.push(cap_type.to_owned());
                if let Some(methods) = entry.get("methods").and_then(serde_json::Value::as_array) {
                    for m in methods {
                        if let Some(method_name) = m.as_str() {
                            caps.push(format!("{cap_type}.{method_name}"));
                        }
                    }
                }
            }
        }
        if !caps.is_empty() {
            generate_semantic_aliases(&mut caps);
            return caps;
        }
    }

    let extracted = extract_from_lifecycle(value);
    if !extracted.is_empty() {
        return extracted;
    }

    caps
}

/// Extract capabilities from a `lifecycle.status` or `capability.list` response.
///
/// Handles Formats A–D and an optional `{"result": ...}` wrapper.
#[must_use]
pub fn extract_from_lifecycle(result: &serde_json::Value) -> Vec<String> {
    let target = result
        .get("result")
        .and_then(|r| r.get("capabilities"))
        .or_else(|| result.get("capabilities"));

    match target {
        Some(serde_json::Value::Array(arr)) => extract_from_array(arr),
        Some(serde_json::Value::Object(obj)) => obj
            .get("capabilities")
            .and_then(serde_json::Value::as_array)
            .map(|arr| extract_from_array(arr))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Extract capability strings from an array that may contain strings
/// (Format A) or objects with a `"name"` field (Format B).
#[must_use]
pub fn extract_from_array(arr: &[serde_json::Value]) -> Vec<String> {
    arr.iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(obj) => obj
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            _ => None,
        })
        .collect()
}

/// Generate well-known semantic aliases from capability types.
///
/// When a primal advertises `crypto` with method-level capabilities like
/// `crypto.blake3_hash`, also register `crypto.hash` since the primal
/// implements the generic hash dispatcher.
pub fn generate_semantic_aliases(caps: &mut Vec<String>) {
    let has = |name: &str, list: &[String]| list.iter().any(|c| c == name);

    let snapshot = caps.clone();
    let mut additions = Vec::new();

    if has("crypto", &snapshot) && !has("crypto.hash", &snapshot) {
        additions.push("crypto.hash".to_owned());
    }
    if has("crypto", &snapshot)
        && !has("crypto.encrypt", &snapshot)
        && has("crypto.chacha20_poly1305_encrypt", &snapshot)
    {
        additions.push("crypto.encrypt".to_owned());
    }
    if has("crypto", &snapshot)
        && !has("crypto.decrypt", &snapshot)
        && has("crypto.chacha20_poly1305_decrypt", &snapshot)
    {
        additions.push("crypto.decrypt".to_owned());
    }
    if has("crypto", &snapshot)
        && !has("crypto.sign", &snapshot)
        && has("crypto.sign_ed25519", &snapshot)
    {
        additions.push("crypto.sign".to_owned());
    }
    if has("crypto", &snapshot)
        && !has("crypto.verify", &snapshot)
        && has("crypto.verify_ed25519", &snapshot)
    {
        additions.push("crypto.verify".to_owned());
    }

    caps.extend(additions);
}

/// Auto-register base capabilities that any responsive primal should have.
pub fn inject_base_capabilities(caps: &mut Vec<String>) {
    for base in ["system.ping", "health.check", "health.liveness"] {
        if !caps.iter().any(|c| c == base) {
            caps.push(base.to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_a_flat_array() {
        let result = serde_json::json!({
            "name": "test",
            "capabilities": ["shader.compile", "shader.compile.wgsl"]
        });
        let caps = extract_from_lifecycle(&result);
        assert_eq!(caps, vec!["shader.compile", "shader.compile.wgsl"]);
    }

    #[test]
    fn format_b_object_array() {
        let result = serde_json::json!({
            "capabilities": [
                {"name": "health", "version": "1.0"},
                {"name": "compute.dispatch"}
            ]
        });
        let caps = extract_from_lifecycle(&result);
        assert_eq!(caps, vec!["health", "compute.dispatch"]);
    }

    #[test]
    fn format_c_nested_wrapper() {
        let result = serde_json::json!({
            "result": {
                "capabilities": ["dag.session.create", "dag.event.append"]
            }
        });
        let caps = extract_from_lifecycle(&result);
        assert_eq!(caps, vec!["dag.session.create", "dag.event.append"]);
    }

    #[test]
    fn format_d_double_nested() {
        let result = serde_json::json!({
            "name": "test",
            "capabilities": {
                "capabilities": ["compute.submit", "compute.status"]
            }
        });
        let caps = extract_from_lifecycle(&result);
        assert_eq!(caps, vec!["compute.submit", "compute.status"]);
    }

    #[test]
    fn missing_returns_empty() {
        let result = serde_json::json!({"name": "test"});
        assert!(extract_from_lifecycle(&result).is_empty());
    }

    #[test]
    fn null_returns_empty() {
        let result = serde_json::json!({"capabilities": null});
        assert!(extract_from_lifecycle(&result).is_empty());
    }

    #[test]
    fn mixed_array_ignores_non_string() {
        let result = serde_json::json!({
            "capabilities": ["valid", 42, null, {"name": "also_valid"}]
        });
        let caps = extract_from_lifecycle(&result);
        assert_eq!(caps, vec!["valid", "also_valid"]);
    }

    #[test]
    fn format_e_beardog_provided_capabilities() {
        let result = serde_json::json!({
            "provided_capabilities": [
                {
                    "type": "crypto",
                    "version": "1.0",
                    "methods": ["blake3_hash", "hmac_sha256", "chacha20_poly1305_encrypt",
                                "chacha20_poly1305_decrypt", "sign_ed25519", "verify_ed25519"]
                },
                {
                    "type": "security",
                    "methods": ["evaluate", "lineage"]
                }
            ]
        });
        let caps = extract_from_any(&result);
        assert!(caps.contains(&"crypto".to_owned()));
        assert!(caps.contains(&"crypto.blake3_hash".to_owned()));
        assert!(
            caps.contains(&"crypto.hash".to_owned()),
            "semantic alias missing"
        );
        assert!(
            caps.contains(&"crypto.encrypt".to_owned()),
            "semantic alias missing"
        );
        assert!(
            caps.contains(&"crypto.sign".to_owned()),
            "semantic alias missing"
        );
        assert!(caps.contains(&"security".to_owned()));
        assert!(caps.contains(&"security.evaluate".to_owned()));
    }

    #[test]
    fn format_f_songbird_flat_array() {
        let result = serde_json::json!([
            "network.discovery",
            "network.federation",
            "ipc.jsonrpc",
            "crypto.delegate"
        ]);
        let caps = extract_from_any(&result);
        assert_eq!(
            caps,
            vec![
                "network.discovery",
                "network.federation",
                "ipc.jsonrpc",
                "crypto.delegate"
            ]
        );
    }

    #[test]
    fn inject_base_adds_required_caps() {
        let mut caps = vec!["crypto".to_owned()];
        inject_base_capabilities(&mut caps);
        assert!(caps.contains(&"system.ping".to_owned()));
        assert!(caps.contains(&"health.check".to_owned()));
        assert!(caps.contains(&"health.liveness".to_owned()));
    }

    #[test]
    fn inject_base_does_not_duplicate() {
        let mut caps = vec!["system.ping".to_owned(), "health.check".to_owned()];
        inject_base_capabilities(&mut caps);
        assert_eq!(
            caps.iter().filter(|c| c.as_str() == "system.ping").count(),
            1
        );
    }
}
