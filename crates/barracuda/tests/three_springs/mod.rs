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
