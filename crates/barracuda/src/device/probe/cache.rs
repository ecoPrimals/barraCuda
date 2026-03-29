// SPDX-License-Identifier: AGPL-3.0-or-later
//! Probe result cache and adapter keying
//!
//! Results are cached globally per adapter identity (name + backend + vendor)
//! so repeated calls are instant.

use crate::device::WgpuDevice;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, MutexGuard};

use super::capabilities::F64BuiltinCapabilities;

/// Acquire a mutex lock, recovering from poison by taking the poisoned data.
///
/// Poison occurs only when another thread panicked while holding the lock.
/// Recovering is correct here because the cache data is always consistent
/// (insertions are atomic) — a poisoned lock just means the inserting thread
/// panicked after writing, which is safe to read.
pub(crate) fn lock_cache<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Global probe result cache keyed by `adapter_name:backend:vendor`
static F64_CAPS_CACHE: LazyLock<Mutex<HashMap<String, F64BuiltinCapabilities>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Legacy single-function cache for backwards compat
static F64_EXP_PROBE_CACHE: LazyLock<Mutex<HashMap<String, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Unique key for caching probe results per physical adapter
pub(crate) fn adapter_key(device: &WgpuDevice) -> String {
    let info = device.adapter_info();
    format!("{}:{:?}:{}", info.name, info.backend, info.vendor)
}

/// Read cached full capability result, if available.
#[must_use]
pub fn cached_f64_builtins(device: &WgpuDevice) -> Option<F64BuiltinCapabilities> {
    lock_cache(&F64_CAPS_CACHE)
        .get(&adapter_key(device))
        .copied()
}

/// Read the cached probe result for this device (legacy single-function API).
#[must_use]
pub fn cached_probe_result(device: &WgpuDevice) -> Option<bool> {
    let key = adapter_key(device);
    if let Some(caps) = lock_cache(&F64_CAPS_CACHE).get(&key).copied() {
        return Some(caps.exp);
    }
    lock_cache(&F64_EXP_PROBE_CACHE).get(&key).copied()
}

/// Pre-populate probe cache from device heuristics before any GPU dispatch.
///
/// Call this immediately after device creation to prime the cache without
/// waiting for an async probe. The async probe overrides this when run.
///
/// Uses `DeviceCapabilities` for vendor-agnostic heuristic seeding.
/// Note: `DeviceCapabilities` is constructed before the cache lock to avoid
/// deadlock (it reads the cache internally via `cached_f64_builtins`).
pub fn seed_cache_from_heuristics(device: &WgpuDevice) {
    let key = adapter_key(device);
    let caps = crate::device::capabilities::DeviceCapabilities::from_device(device);
    let heuristic = F64BuiltinCapabilities {
        basic_f64: true,
        exp: caps.has_reliable_f64(),
        log: caps.has_reliable_f64(),
        exp2: caps.has_reliable_f64(),
        log2: caps.has_reliable_f64(),
        sin: !caps.needs_sin_f64_workaround(),
        cos: !caps.needs_cos_f64_workaround(),
        sqrt: true,
        fma: true,
        abs_min_max: true,
        composite_transcendental: false,
        exp_log_chain: false,
        shared_mem_f64: !device.is_nvk(),
        df64_arith: true,
        df64_transcendentals_safe: caps.df64_transcendentals_safe(),
        df64_fma_two_prod: true,
        df64_workgroup_reduce: true,
    };
    let mut cache = lock_cache(&F64_CAPS_CACHE);
    cache.entry(key.clone()).or_insert(heuristic);
    let exp_capable = cache.get(&key).is_some_and(|c| c.exp);
    drop(cache);
    lock_cache(&F64_EXP_PROBE_CACHE)
        .entry(key)
        .or_insert(exp_capable);
}

/// Check if the cached probe result for this adapter key indicates basic f64 works.
///
/// Returns `Some(true)` if probed and working, `Some(false)` if probed and broken,
/// `None` if not yet probed.
#[must_use]
pub fn cached_basic_f64_for_key(key: &str) -> Option<bool> {
    lock_cache(&F64_CAPS_CACHE).get(key).map(|c| c.basic_f64)
}

/// Check if the cached probe indicates `var<workgroup>` f64 reductions work.
///
/// Returns `Some(true)` if shared-memory f64 reductions pass the probe,
/// `Some(false)` if they produce zeros, `None` if not yet probed.
#[must_use]
pub fn cached_shared_mem_f64_for_key(key: &str) -> Option<bool> {
    lock_cache(&F64_CAPS_CACHE)
        .get(key)
        .map(|c| c.shared_mem_f64)
}

pub(super) fn insert_full_caps(key: String, caps: F64BuiltinCapabilities) {
    lock_cache(&F64_EXP_PROBE_CACHE).insert(key.clone(), caps.exp);
    lock_cache(&F64_CAPS_CACHE).insert(key, caps);
}

/// Seeds the global probe cache for unit tests (synthetic adapter keys).
#[cfg(test)]
#[expect(dead_code, reason = "available for future probe-aware tests")]
pub(crate) fn insert_caps_for_test(key: String, caps: F64BuiltinCapabilities) {
    insert_full_caps(key, caps);
}

pub(super) fn get_cached_full(key: &str) -> Option<F64BuiltinCapabilities> {
    lock_cache(&F64_CAPS_CACHE).get(key).copied()
}

pub(super) fn get_cached_exp(key: &str) -> Option<bool> {
    lock_cache(&F64_CAPS_CACHE)
        .get(key)
        .copied()
        .map(|c| c.exp)
        .or_else(|| lock_cache(&F64_EXP_PROBE_CACHE).get(key).copied())
}

pub(super) fn insert_exp_only(key: String, capable: bool) {
    lock_cache(&F64_EXP_PROBE_CACHE).insert(key, capable);
}
