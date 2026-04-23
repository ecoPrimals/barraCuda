// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for `SovereignDevice` — capability-based IPC dispatch backend.

use super::*;
#[cfg(feature = "sovereign-dispatch")]
use super::super::sovereign_discovery::{
    DISPATCH_ADDR_ENV, DISPATCH_CAPABILITY, detect_dispatch_addr,
};

#[cfg(feature = "sovereign-dispatch")]
use std::collections::HashMap;
#[cfg(feature = "sovereign-dispatch")]
use std::sync::Arc;

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn device_creation_ipc() {
    let dev = SovereignDevice::new();
    assert!(dev.name().contains("sovereign"));
    assert!(dev.has_f64_shaders());
    assert!(!dev.is_lost());
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn default_matches_new() {
    let dev = SovereignDevice::default();
    assert!(dev.name().contains("sovereign"));
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn dispatch_binary_without_toadstool() {
    let dev = SovereignDevice::new();
    let result = dev.dispatch_binary(&[0xDE, 0xAD], vec![], (1, 1, 1), "main");
    assert!(result.is_err());
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn buffer_alloc_stages_data() {
    let dev = SovereignDevice::new();
    let buf = dev.alloc_buffer("test", 64).unwrap();
    assert_eq!(buf.size, 64);

    dev.upload(&buf, 0, &[1, 2, 3, 4]);
    let data = dev.download(&buf, 64).unwrap();
    assert_eq!(data[..4], [1, 2, 3, 4]);
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn buffer_alloc_assigns_unique_ids() {
    let dev = SovereignDevice::new();
    let b1 = dev.alloc_buffer("a", 1024).unwrap();
    let b2 = dev.alloc_buffer("b", 2048).unwrap();
    assert_ne!(b1.id, b2.id);
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn dispatch_env_constant_is_correct() {
    assert_eq!(DISPATCH_ADDR_ENV, "BARRACUDA_DISPATCH_ADDR");
    assert_eq!(DISPATCH_CAPABILITY, "compute.dispatch");
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn detect_dispatch_addr_graceful_without_toadstool() {
    let addr = detect_dispatch_addr();
    if let Some(ref a) = addr {
        assert!(!a.is_empty(), "discovered address must be non-empty");
    }
}

#[test]
fn coral_cache_lookup_returns_none_for_unknown_shader() {
    let result = SovereignDevice::try_coral_cache("nonexistent_shader_source_12345");
    assert!(result.is_none());
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn select_target_single_arch() {
    let dev = SovereignDevice::new();
    let archs = vec!["sm_75".to_string()];
    let result = dev.select_target(&archs).unwrap();
    assert_eq!(result, "sm_75");
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn select_target_empty_archs() {
    let dev = SovereignDevice::new();
    let archs: Vec<String> = vec![];
    assert!(dev.select_target(&archs).is_err());
}

#[cfg(feature = "sovereign-dispatch")]
#[test]
fn query_dispatch_arch_no_endpoint() {
    let dev = SovereignDevice {
        name: Arc::from("test"),
        compiler_available: false,
        dispatch_addr: None,
        binary_cache: std::sync::Mutex::new(HashMap::new()),
        staged_buffers: std::sync::Mutex::new(HashMap::new()),
    };
    assert!(dev.query_dispatch_arch().is_none());
}
