// SPDX-License-Identifier: AGPL-3.0-only
//! Shared helpers for Three Springs Evolution tests.
//!
//! These primitives serve wetSpring, airSpring, and hotSpring validation.

use barracuda::device::WgpuDevice;
use std::sync::Arc;

pub fn create_device_sync() -> Option<Arc<WgpuDevice>> {
    barracuda::device::test_pool::tokio_block_on(async {
        match WgpuDevice::new_f64_capable().await {
            Ok(d) => Some(Arc::new(d)),
            Err(_) => None,
        }
    })
}

/// Tolerance appropriate for the device: tight on known-good f64 hardware,
/// relaxed when the GPU claims f64 capability but achieves limited precision
/// (NVK, DF64 emulation, some integrated GPUs).
///
/// Software adapters are already excluded by `new_f64_capable()` which only
/// returns `DiscreteGpu` or `IntegratedGpu` device types.
pub fn tol(device: &WgpuDevice, hardware_tol: f64) -> f64 {
    if barracuda::device::test_harness::is_software_adapter(device) {
        hardware_tol.max(1e-4)
    } else {
        // Real GPU with f64 shaders may still use DF64 emulation or have
        // driver-specific precision limits (NVK, nouveau, some AMD).
        // Floor at 1e-6 to avoid false failures on valid hardware.
        hardware_tol.max(1e-6)
    }
}

/// CPU reference Shannon entropy
pub fn cpu_shannon(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    counts
        .iter()
        .map(|&c| {
            let p = c / total;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum()
}

/// CPU reference Simpson index
pub fn cpu_simpson(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    counts.iter().map(|&c| (c / total).powi(2)).sum()
}

mod basic_tests;
mod e2e_tests;
mod edge_case_tests;
mod precision_tests;
