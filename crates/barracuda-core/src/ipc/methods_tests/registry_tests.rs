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
fn registered_methods_count() {
    assert_eq!(REGISTERED_METHODS.len(), 58);
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
