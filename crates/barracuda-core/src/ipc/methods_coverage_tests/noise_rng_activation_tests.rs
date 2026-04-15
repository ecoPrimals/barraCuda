// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for noise.*, rng.*, and activation.* handlers.

use crate::ipc::jsonrpc;

use super::super::math::{
    activation_fitts, activation_hick, noise_perlin2d, noise_perlin3d, rng_uniform,
};

// ── noise.perlin2d ──────────────────────────────────────────────────

#[test]
fn noise_perlin2d_missing_x() {
    let resp = noise_perlin2d(&serde_json::json!({"y": 1.0}), serde_json::json!(113));
    assert!(resp.error.is_some());
}

#[test]
fn noise_perlin2d_missing_y() {
    let resp = noise_perlin2d(&serde_json::json!({"x": 1.0}), serde_json::json!(114));
    assert!(resp.error.is_some());
}

#[test]
fn noise_perlin2d_happy_path() {
    let resp = noise_perlin2d(
        &serde_json::json!({"x": 0.5, "y": 0.5}),
        serde_json::json!(115),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!(result.is_finite());
}

// ── noise.perlin3d ──────────────────────────────────────────────────

#[test]
fn noise_perlin3d_missing_z() {
    let resp = noise_perlin3d(
        &serde_json::json!({"x": 1.0, "y": 1.0}),
        serde_json::json!(116),
    );
    assert!(resp.error.is_some());
}

#[test]
fn noise_perlin3d_happy_path() {
    let resp = noise_perlin3d(
        &serde_json::json!({"x": 0.5, "y": 0.5, "z": 0.5}),
        serde_json::json!(117),
    );
    assert!(resp.error.is_none());
}

#[test]
fn noise_perlin3d_missing_x_only() {
    let resp = noise_perlin3d(
        &serde_json::json!({"y": 1.0, "z": 2.0}),
        serde_json::json!(602),
    );
    let err = resp.error.expect("missing x should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains('x'));
}

#[test]
fn noise_perlin3d_missing_y_only() {
    let resp = noise_perlin3d(
        &serde_json::json!({"x": 1.0, "z": 2.0}),
        serde_json::json!(603),
    );
    let err = resp.error.expect("missing y should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains('y'));
}

// ── rng.uniform ─────────────────────────────────────────────────────

#[test]
fn rng_uniform_max_lte_min() {
    let resp = rng_uniform(
        &serde_json::json!({"min": 5.0, "max": 3.0}),
        serde_json::json!(118),
    );
    let err = resp.error.expect("max <= min should fail");
    assert!(err.message.contains("max must be > min"));
}

#[test]
fn rng_uniform_happy_path() {
    let resp = rng_uniform(
        &serde_json::json!({"n": 5, "min": 0.0, "max": 1.0, "seed": 42}),
        serde_json::json!(119),
    );
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(arr.len(), 5);
    for v in &arr {
        let f = v.as_f64().unwrap();
        assert!((0.0..1.0).contains(&f));
    }
}

#[test]
fn rng_uniform_defaults() {
    let resp = rng_uniform(&serde_json::json!({}), serde_json::json!(120));
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(arr.len(), 1);
}

// ── activation.fitts ────────────────────────────────────────────────

#[test]
fn activation_fitts_missing_distance() {
    let resp = activation_fitts(&serde_json::json!({"width": 10.0}), serde_json::json!(121));
    assert!(resp.error.is_some());
}

#[test]
fn activation_fitts_missing_width() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0}),
        serde_json::json!(122),
    );
    assert!(resp.error.is_some());
}

#[test]
fn activation_fitts_zero_width() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 0.0}),
        serde_json::json!(123),
    );
    let err = resp.error.expect("zero width should fail");
    assert!(err.message.contains("width must be > 0"));
}

#[test]
fn activation_fitts_unknown_variant() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 10.0, "variant": "unknown"}),
        serde_json::json!(124),
    );
    let err = resp.error.expect("unknown variant should fail");
    assert!(err.message.contains("Unknown variant"));
}

#[test]
fn activation_fitts_shannon_happy_path() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 10.0}),
        serde_json::json!(125),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert!(r["movement_time"].as_f64().unwrap() > 0.0);
    assert_eq!(r["variant"], "shannon");
}

#[test]
fn activation_fitts_original_variant() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 10.0, "variant": "fitts"}),
        serde_json::json!(126),
    );
    assert!(resp.error.is_none());
    assert_eq!(resp.result.unwrap()["variant"], "fitts");
}

// ── activation.hick ─────────────────────────────────────────────────

#[test]
fn activation_hick_missing_n_choices() {
    let resp = activation_hick(&serde_json::json!({}), serde_json::json!(127));
    assert!(resp.error.is_some());
}

#[test]
fn activation_hick_zero_choices() {
    let resp = activation_hick(&serde_json::json!({"n_choices": 0}), serde_json::json!(128));
    let err = resp.error.expect("zero choices should fail");
    assert!(err.message.contains("n_choices must be > 0"));
}

#[test]
fn activation_hick_happy_path() {
    let resp = activation_hick(&serde_json::json!({"n_choices": 8}), serde_json::json!(129));
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let bits = r["information_bits"].as_f64().unwrap();
    assert!((bits - 3.0).abs() < 1e-10);
}

#[test]
fn activation_hick_with_no_choice() {
    let resp = activation_hick(
        &serde_json::json!({"n_choices": 4, "include_no_choice": true}),
        serde_json::json!(130),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert!(r["include_no_choice"].as_bool().unwrap());
    let bits = r["information_bits"].as_f64().unwrap();
    assert!((bits - 5.0_f64.log2()).abs() < 1e-10);
}
