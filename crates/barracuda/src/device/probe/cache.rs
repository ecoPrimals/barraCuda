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
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Global probe result cache keyed by adapter_name:backend:vendor
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
pub fn cached_f64_builtins(device: &WgpuDevice) -> Option<F64BuiltinCapabilities> {
    lock_cache(&F64_CAPS_CACHE)
        .get(&adapter_key(device))
        .copied()
}

/// Read the cached probe result for this device (legacy single-function API).
pub fn cached_probe_result(device: &WgpuDevice) -> Option<bool> {
    let key = adapter_key(device);
    if let Some(caps) = lock_cache(&F64_CAPS_CACHE).get(&key).copied() {
        return Some(caps.exp);
    }
    lock_cache(&F64_EXP_PROBE_CACHE).get(&key).copied()
}

/// Pre-populate probe cache from device name heuristics before any GPU dispatch.
///
/// Call this immediately after device creation to prime the cache without
/// waiting for an async probe. The async probe overrides this when run.
pub fn seed_cache_from_heuristics(device: &WgpuDevice) {
    let key = adapter_key(device);
    let mut cache = lock_cache(&F64_CAPS_CACHE);
    cache.entry(key.clone()).or_insert_with(|| {
        let exp_log_works = !device.needs_f64_exp_log_workaround();
        F64BuiltinCapabilities {
            basic_f64: true,
            exp: exp_log_works,
            log: exp_log_works,
            exp2: exp_log_works,
            log2: exp_log_works,
            sin: exp_log_works,
            cos: exp_log_works,
            sqrt: true,
            fma: true,
            abs_min_max: true,
        }
    });
    let exp_capable = cache.get(&key).is_some_and(|c| c.exp);
    drop(cache);
    lock_cache(&F64_EXP_PROBE_CACHE)
        .entry(key)
        .or_insert(exp_capable);
}

pub(super) fn insert_full_caps(key: String, caps: F64BuiltinCapabilities) {
    lock_cache(&F64_CAPS_CACHE).insert(key.clone(), caps);
    lock_cache(&F64_EXP_PROBE_CACHE).insert(key, caps.exp);
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
