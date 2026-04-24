// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for BTSP Phase 2+3 handshake guard and relay helpers.

use super::*;

#[test]
fn btsp_outcome_dev_mode_accepts() {
    let outcome = BtspOutcome::DevMode;
    assert!(outcome.should_accept());
}

#[test]
fn btsp_outcome_authenticated_accepts() {
    let outcome = BtspOutcome::Authenticated(BtspSession {
        session_id: "test-session".to_string(),
        cipher: BtspCipher::Null,
        session_key: Vec::new(),
    });
    assert!(outcome.should_accept());
    assert!(outcome.session().is_some());
}

#[test]
fn btsp_outcome_degraded_accepts() {
    let outcome = BtspOutcome::Degraded {
        reason: "security provider not found".to_string(),
        consumed_line: None,
    };
    assert!(outcome.should_accept());
    assert!(outcome.consumed_line().is_none());
}

#[test]
fn btsp_outcome_degraded_with_consumed_line() {
    let outcome = BtspOutcome::Degraded {
        reason: "legacy client".to_string(),
        consumed_line: Some(r#"{"jsonrpc":"2.0","method":"tensor.dot","id":1}"#.to_string()),
    };
    assert!(outcome.should_accept());
    assert!(outcome.consumed_line().unwrap().contains("tensor.dot"));
}

#[test]
fn btsp_outcome_rejected_refuses() {
    let outcome = BtspOutcome::Rejected {
        reason: "bad challenge response".to_string(),
    };
    assert!(!outcome.should_accept());
}

#[test]
fn discover_security_provider_returns_none_when_no_socket() {
    assert!(discover_security_provider().is_none());
}

#[test]
fn btsp_outcome_debug_impl() {
    let outcome = BtspOutcome::Authenticated(BtspSession {
        session_id: "abc".into(),
        cipher: BtspCipher::ChaCha20Poly1305,
        session_key: vec![0u8; 32],
    });
    let debug = format!("{outcome:?}");
    assert!(debug.contains("Authenticated"));
    assert!(debug.contains("abc"));
}

#[test]
fn btsp_cipher_from_wire_variants() {
    assert_eq!(
        BtspCipher::from_wire("chacha20_poly1305"),
        BtspCipher::ChaCha20Poly1305
    );
    assert_eq!(
        BtspCipher::from_wire("chacha20"),
        BtspCipher::ChaCha20Poly1305
    );
    assert_eq!(BtspCipher::from_wire("hmac_plain"), BtspCipher::HmacPlain);
    assert_eq!(BtspCipher::from_wire("hmac"), BtspCipher::HmacPlain);
    assert_eq!(BtspCipher::from_wire("none"), BtspCipher::Null);
    assert_eq!(BtspCipher::from_wire("null"), BtspCipher::Null);
    assert_eq!(BtspCipher::from_wire(""), BtspCipher::Null);
}

#[test]
fn btsp_cipher_requires_key() {
    assert!(!BtspCipher::Null.requires_key());
    assert!(BtspCipher::HmacPlain.requires_key());
    assert!(BtspCipher::ChaCha20Poly1305.requires_key());
}

#[test]
fn is_btsp_client_hello_type_field() {
    let msg: serde_json::Value =
        serde_json::json!({"type": "ClientHello", "client_ephemeral_pub": "abc"});
    assert!(is_btsp_client_hello(&msg));
}

#[test]
fn is_btsp_client_hello_protocol_field() {
    let msg: serde_json::Value =
        serde_json::json!({"protocol": "btsp", "version": 1, "client_ephemeral_pub": "abc"});
    assert!(is_btsp_client_hello(&msg));
}

#[test]
fn is_btsp_client_hello_rejects_plain_jsonrpc() {
    let msg: serde_json::Value =
        serde_json::json!({"jsonrpc": "2.0", "method": "tensor.dot", "id": 1});
    assert!(!is_btsp_client_hello(&msg));
}

#[test]
fn hex_to_bytes_valid() {
    let bytes = hex_to_bytes("deadbeef").unwrap();
    assert_eq!(bytes, vec![0xde, 0xad, 0xbe, 0xef]);
}

#[test]
fn hex_to_bytes_odd_length() {
    assert!(hex_to_bytes("abc").is_none());
}

#[test]
fn hex_to_bytes_invalid_chars() {
    assert!(hex_to_bytes("zzzz").is_none());
}

#[test]
fn hex_to_bytes_empty() {
    assert_eq!(hex_to_bytes("").unwrap(), Vec::<u8>::new());
}

#[test]
fn resolve_family_seed_raw_returns_none_when_unset() {
    if std::env::var("FAMILY_SEED").is_ok()
        || std::env::var("BEARDOG_FAMILY_SEED").is_ok()
        || std::env::var("BIOMEOS_FAMILY_SEED").is_ok()
    {
        return;
    }
    assert!(resolve_family_seed_raw().is_none());
}
