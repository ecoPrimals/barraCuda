// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for parse_shape, normalize_method, and REGISTERED_METHODS.

use super::*;

#[test]
fn parse_shape_valid() {
    let arr = vec![
        serde_json::json!(2),
        serde_json::json!(3),
        serde_json::json!(4),
    ];
    let shape = parse_shape(&arr).expect("valid shape");
    assert_eq!(shape, vec![2, 3, 4]);
}

#[test]
fn parse_shape_single_element() {
    let arr = vec![serde_json::json!(128)];
    assert_eq!(parse_shape(&arr), Some(vec![128]));
}

#[test]
fn parse_shape_empty() {
    let arr: Vec<serde_json::Value> = vec![];
    assert_eq!(parse_shape(&arr), Some(vec![]));
}

#[test]
fn parse_shape_with_non_numeric() {
    let arr = vec![serde_json::json!(2), serde_json::json!("bad")];
    let shape = parse_shape(&arr);
    assert!(
        shape.is_none() || shape.as_ref().is_some_and(|s| s.len() < 2),
        "non-numeric values should be filtered out"
    );
}

#[test]
fn normalize_strips_legacy_namespace() {
    let legacy = format!("{}.device.list", crate::PRIMAL_NAMESPACE);
    assert_eq!(normalize_method(&legacy), "device.list");
}

#[test]
fn normalize_passes_through_standard_names() {
    assert_eq!(normalize_method("device.list"), "device.list");
    assert_eq!(normalize_method("health.check"), "health.check");
}

#[test]
fn normalize_passes_through_foreign_prefix() {
    assert_eq!(
        normalize_method("other_primal.device.list"),
        "other_primal.device.list"
    );
}

#[test]
fn normalize_empty() {
    assert_eq!(normalize_method(""), "");
}

#[test]
fn registered_methods_count_nonzero_and_unique() {
    assert!(
        REGISTERED_METHODS.len() >= 70,
        "sanity: expected at least 70 methods, got {}",
        REGISTERED_METHODS.len()
    );
    let mut seen = std::collections::HashSet::new();
    for method in REGISTERED_METHODS {
        assert!(
            seen.insert(method),
            "duplicate method in REGISTERED_METHODS: {method}"
        );
    }
}

#[test]
fn registered_methods_semantic_format() {
    for method in REGISTERED_METHODS {
        assert!(
            method.contains('.'),
            "method {method} should use domain.operation format"
        );
        assert!(
            !method.starts_with(&format!("{}.", crate::PRIMAL_NAMESPACE)),
            "method {method} should NOT have primal namespace prefix"
        );
    }
}

#[test]
fn normalize_just_namespace_no_dot() {
    assert_eq!(
        normalize_method(crate::PRIMAL_NAMESPACE),
        crate::PRIMAL_NAMESPACE
    );
}

#[test]
fn normalize_namespace_with_dot() {
    let input = format!("{}.", crate::PRIMAL_NAMESPACE);
    assert_eq!(normalize_method(&input), "");
}

#[test]
fn normalize_legacy_prefix_accepted() {
    let legacy = format!("{}.device.list", crate::PRIMAL_NAMESPACE);
    let resp_method = normalize_method(&legacy);
    assert_eq!(resp_method, "device.list");
}

/// Vertebrate self-audit: every REGISTERED_METHOD must appear in the dispatch
/// match arms. If a method is registered but not dispatched, consumers will get
/// `METHOD_NOT_FOUND` — exactly the phantom API pattern westGate exposed.
#[tokio::test]
async fn every_registered_method_dispatches() {
    let (primal, _guard) = crate::test_util::start_primal_guarded().await;

    for method in REGISTERED_METHODS {
        let resp = dispatch(
            &primal,
            method,
            &serde_json::json!({}),
            serde_json::json!(1),
        )
        .await;
        let error_code = resp
            .error
            .as_ref()
            .map(|e| e.code);
        assert_ne!(
            error_code,
            Some(-32601), // METHOD_NOT_FOUND
            "Registered method {method} returned METHOD_NOT_FOUND — phantom API!"
        );
    }
}

/// Every REGISTERED_METHOD must also exist in capability_registry.toml
/// (or be a documented alias). Prevents silent divergence.
#[test]
fn registry_toml_covers_registered_methods() {
    let toml_content =
        include_str!("../../../../../config/capability_registry.toml");

    for method in REGISTERED_METHODS {
        // Aliases are documented in the TOML as aliases = { "stats.eigh" = ... }
        // so searching for the quoted method name covers both methods and aliases.
        let quoted = format!("\"{method}\"");
        let bare_name = method.rsplit('.').next().unwrap_or(method);
        assert!(
            toml_content.contains(&quoted) || toml_content.contains(bare_name),
            "REGISTERED_METHOD {method} not found in capability_registry.toml"
        );
    }
}
