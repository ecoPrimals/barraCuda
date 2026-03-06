// SPDX-License-Identifier: AGPL-3.0-or-later
//! Types and configuration for multi-GPU workload distribution.

use super::topology::{GpuDriver, GpuVendor};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Workload distribution configuration.
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Maximum parallel tasks across devices.
    pub max_parallel: usize,
    /// Prefer discrete GPUs over integrated.
    pub prefer_discrete: bool,
    /// Exclude software renderer (e.g. llvmpipe).
    pub exclude_software: bool,
    /// Minimum GFLOPS to include a device.
    pub min_gflops: f64,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            max_parallel: 4,
            prefer_discrete: true,
            exclude_software: true,
            min_gflops: 100.0,
        }
    }
}

/// Device selection requirements for advanced pool acquisition.
#[derive(Debug, Clone, Default)]
pub struct DeviceRequirements {
    /// Minimum VRAM in bytes.
    pub min_vram_bytes: Option<u64>,
    /// Preferred GPU vendor.
    pub preferred_vendor: Option<GpuVendor>,
    /// Exclude software renderer.
    pub exclude_software: bool,
    /// Require discrete GPU.
    pub require_discrete: bool,
    /// Minimum GFLOPS.
    pub min_gflops: Option<f64>,
}

impl DeviceRequirements {
    /// Create requirements with software excluded by default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            exclude_software: true,
            ..Self::default()
        }
    }

    /// Require minimum VRAM in bytes.
    #[must_use]
    pub fn with_min_vram_bytes(mut self, bytes: u64) -> Self {
        self.min_vram_bytes = Some(bytes);
        self
    }

    /// Require minimum VRAM in gigabytes.
    #[must_use]
    pub fn with_min_vram_gb(self, gb: u64) -> Self {
        self.with_min_vram_bytes(gb * 1024 * 1024 * 1024)
    }

    /// Prefer NVIDIA GPUs.
    #[must_use]
    pub fn prefer_nvidia(mut self) -> Self {
        self.preferred_vendor = Some(GpuVendor::Nvidia);
        self
    }

    /// Prefer AMD GPUs.
    #[must_use]
    pub fn prefer_amd(mut self) -> Self {
        self.preferred_vendor = Some(GpuVendor::Amd);
        self
    }

    /// Require discrete (non-integrated) GPU.
    #[must_use]
    pub fn require_discrete(mut self) -> Self {
        self.require_discrete = true;
        self
    }

    /// Require minimum GFLOPS.
    #[must_use]
    pub fn with_min_gflops(mut self, gflops: f64) -> Self {
        self.min_gflops = Some(gflops);
        self
    }

    /// Score a device against these requirements; `None` if device doesn't qualify.
    pub(crate) fn score(&self, info: &DeviceInfo) -> Option<i64> {
        if self.exclude_software && info.vendor == GpuVendor::Software {
            return None;
        }
        if self.require_discrete && !info.is_discrete {
            return None;
        }
        if let Some(min_vram) = self.min_vram_bytes {
            if info.vram_bytes < min_vram {
                return None;
            }
        }
        if let Some(min_gflops) = self.min_gflops {
            if info.estimated_gflops < min_gflops {
                return None;
            }
        }

        let mut score: i64 = 0;
        if let Some(pref) = self.preferred_vendor {
            if info.vendor == pref {
                score += 1000;
            }
        }
        score += (info.vram_bytes / (1024 * 1024 * 1024)) as i64;
        score += (info.estimated_gflops / 100.0) as i64;
        if info.is_discrete {
            score += 100;
        }
        if !info.is_busy() {
            score += 50;
        }
        Some(score)
    }
}

/// Per-device metadata for the advanced pool.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Adapter index in enumeration.
    pub index: usize,
    pub(crate) pool_index: usize,
    /// Device name (e.g. "NVIDIA `GeForce` RTX 4090").
    pub name: String,
    /// GPU vendor.
    pub vendor: GpuVendor,
    /// Driver type (e.g. Nvk, Cuda).
    pub driver: GpuDriver,
    /// Estimated VRAM in bytes.
    pub vram_bytes: u64,
    /// Estimated FP32 GFLOPS.
    pub estimated_gflops: f64,
    /// True if discrete (not integrated).
    pub is_discrete: bool,
    pub(crate) allocations: Arc<AtomicUsize>,
    pub(crate) allocated_bytes: Arc<AtomicU64>,
    pub(crate) busy: Arc<AtomicBool>,
}

impl DeviceInfo {
    /// Returns true if device is currently in use.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Relaxed)
    }

    /// Number of active buffer allocations.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Total allocated bytes on this device.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Estimated available VRAM (total minus allocated).
    #[must_use]
    pub fn available_vram_bytes(&self) -> u64 {
        self.vram_bytes.saturating_sub(self.allocated_bytes())
    }

    /// VRAM usage as percentage (0–100).
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.vram_bytes == 0 {
            return 0.0;
        }
        (self.allocated_bytes() as f64 / self.vram_bytes as f64) * 100.0
    }

    /// Returns true if native f64 shader builtins are supported.
    #[must_use]
    pub fn supports_f64_builtins(&self) -> bool {
        !matches!(self.driver, GpuDriver::Nvk | GpuDriver::Software)
    }
}
