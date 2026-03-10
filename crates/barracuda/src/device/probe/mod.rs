// SPDX-License-Identifier: AGPL-3.0-only
//! Runtime GPU capability probing for f64 built-in functions
//!
//! Dispatches tiny test shaders to empirically verify hardware capabilities
//! rather than relying on driver name heuristics. Each function is compiled in
//! an isolated shader so a crash in one does not mask others.
//!
//! # Design
//!
//! Name-based detection (`is_nvk`, `is_radv`) is synchronous and fast but fragile.
//! Probe-based detection is async and definitive. Results are cached globally per
//! adapter identity (name + backend + vendor) so repeated calls are instant.
//!
//! The probe unlocks a key insight: WGSL → naga → SPIR-V → Vulkan bypasses the
//! proprietary software FP64 lock that CUDA/OpenCL enforce on consumer cards.
//! Both RTX 3090 (Ampere) and RX 6950 XT (RDNA2) expose `VK_KHR_shader_float64`
//! natively. By probing each builtin individually we build an exact capability
//! matrix, allowing `ShaderTemplate` to use native calls wherever safe and fall
//! back to the `math_f64.wgsl` software library only where needed.
//!
//! # Usage
//!
//! ```rust,ignore
//! let caps = probe_f64_builtins(&device).await;
//! // caps.exp  == true  → native exp(f64(x)) works
//! // caps.exp  == false → use software exp_f64() from math_f64.wgsl
//! ```

pub(crate) mod cache;
mod capabilities;
mod probes;
mod runner;

use crate::device::WgpuDevice;

pub(crate) use cache::{adapter_key, lock_cache};
pub use cache::{
    cached_basic_f64_for_key, cached_f64_builtins, cached_probe_result,
    cached_shared_mem_f64_for_key, seed_cache_from_heuristics,
};
pub use capabilities::F64BuiltinCapabilities;

use cache::{get_cached_exp, get_cached_full, insert_exp_only, insert_full_caps};
use probes::PROBES;
use runner::run_single_probe;

/// Probe ALL f64 built-in functions available on this device.
///
/// Each function is tested in an isolated shader so a crash in one does not
/// hide support for others. Results are cached globally per adapter.
pub async fn probe_f64_builtins(device: &WgpuDevice) -> F64BuiltinCapabilities {
    let key = adapter_key(device);

    if let Some(cached) = get_cached_full(&key) {
        return cached;
    }

    let mut caps = F64BuiltinCapabilities::none();
    for probe in PROBES {
        let ok = run_single_probe(device, probe).await;
        match probe.name {
            "basic_f64" => caps.basic_f64 = ok,
            "exp" => caps.exp = ok,
            "log" => caps.log = ok,
            "exp2" => caps.exp2 = ok,
            "log2" => caps.log2 = ok,
            "sin" => caps.sin = ok,
            "cos" => caps.cos = ok,
            "sqrt" => caps.sqrt = ok,
            "fma" => caps.fma = ok,
            "abs_min_max" => caps.abs_min_max = ok,
            "shared_mem_f64" => caps.shared_mem_f64 = ok,
            _ => {}
        }
        if probe.name == "basic_f64" && !ok {
            tracing::warn!("Device cannot compile basic f64 WGSL — forcing DF64 for all shaders");
            break;
        }
    }

    let profile = crate::device::driver_profile::GpuDriverProfile::from_device(device);
    caps.df64_arith = true;
    caps.df64_transcendentals_safe = !profile.has_nvvm_df64_poisoning_risk();

    insert_full_caps(key, caps);
    caps
}

/// Probe whether this device supports native `exp(f64)` / `log(f64)` (legacy API).
///
/// If the full `probe_f64_builtins` has already been run, reads from that cache.
/// Otherwise runs only the exp probe for speed.
pub async fn probe_f64_exp_capable(device: &WgpuDevice) -> bool {
    let key = adapter_key(device);

    if let Some(caps) = get_cached_full(&key) {
        return caps.exp;
    }

    if let Some(cached) = get_cached_exp(&key) {
        return cached;
    }

    // Run exp probe only (index 1) for legacy single-function API
    let capable = run_single_probe(device, &PROBES[1]).await;
    insert_exp_only(key, capable);
    capable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_and_read_cache() {
        use crate::device::test_pool::get_test_device_sync;
        let dev = get_test_device_sync();
        seed_cache_from_heuristics(&dev);
        let result = cached_probe_result(&dev);
        assert!(
            result.is_some(),
            "Cache should be seeded after seed_cache_from_heuristics"
        );
    }

    #[tokio::test]
    async fn test_probe_returns_consistent_result() {
        use crate::device::test_pool::get_test_device;
        let dev = get_test_device().await;
        let first = probe_f64_exp_capable(&dev).await;
        let second = probe_f64_exp_capable(&dev).await;
        assert_eq!(first, second, "Probe should be deterministic/cached");
    }

    #[tokio::test]
    async fn test_full_caps_probe_consistency() {
        use crate::device::test_pool::get_test_device;
        let dev = get_test_device().await;
        let caps1 = probe_f64_builtins(&dev).await;
        let caps2 = probe_f64_builtins(&dev).await;
        assert_eq!(caps1, caps2, "Full probe should be deterministic/cached");
    }

    #[tokio::test]
    async fn test_caps_exp_agrees_with_single_probe() {
        use crate::device::test_pool::get_test_device;
        let dev = get_test_device().await;
        let caps = probe_f64_builtins(&dev).await;
        let exp_only = probe_f64_exp_capable(&dev).await;
        assert_eq!(
            caps.exp, exp_only,
            "Full caps exp field must agree with single exp probe"
        );
    }

    #[test]
    fn test_f64_caps_none() {
        let c = F64BuiltinCapabilities::none();
        assert_eq!(c.native_count(), 0);
        assert!(!c.can_compile_f64());
        assert!(c.needs_exp_log_workaround());
        assert!(c.needs_sin_f64_workaround());
        assert!(c.needs_cos_f64_workaround());
        assert!(c.needs_shared_mem_f64_workaround());
    }

    #[test]
    fn test_f64_caps_full() {
        let c = F64BuiltinCapabilities::full();
        assert_eq!(c.native_count(), 12);
        assert!(c.can_compile_f64());
        assert!(!c.needs_exp_log_workaround());
        assert!(!c.needs_sin_f64_workaround());
        assert!(!c.needs_cos_f64_workaround());
        assert!(!c.needs_shared_mem_f64_workaround());
        assert!(c.can_use_df64());
        assert!(!c.needs_df64_transcendental_stripping());
    }

    #[test]
    fn test_basic_f64_false_forces_zero_native() {
        let c = F64BuiltinCapabilities {
            basic_f64: false,
            exp: true,
            log: true,
            exp2: true,
            log2: true,
            sin: true,
            cos: true,
            sqrt: true,
            fma: true,
            abs_min_max: true,
            shared_mem_f64: true,
            df64_arith: true,
            df64_transcendentals_safe: true,
        };
        assert_eq!(
            c.native_count(),
            0,
            "basic_f64=false must force native_count to 0"
        );
        assert!(c.needs_exp_log_workaround());
        assert!(c.needs_sin_f64_workaround());
        assert!(c.needs_cos_f64_workaround());
        assert!(c.needs_shared_mem_f64_workaround());
    }

    #[test]
    fn test_df64_poisoning_risk() {
        let mut c = F64BuiltinCapabilities::full();
        c.df64_transcendentals_safe = false;
        assert!(c.needs_df64_transcendental_stripping());
        assert!(c.can_use_df64());
    }

    #[test]
    fn test_no_df64_at_all() {
        let mut c = F64BuiltinCapabilities::full();
        c.df64_arith = false;
        c.df64_transcendentals_safe = false;
        assert!(!c.can_use_df64());
    }
}
