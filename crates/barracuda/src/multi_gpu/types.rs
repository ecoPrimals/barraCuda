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

        const PREFERRED_VENDOR_BONUS: i64 = 1000;
        const DISCRETE_BONUS: i64 = 100;
        const IDLE_BONUS: i64 = 50;
        const GFLOPS_DIVISOR: f64 = 100.0;

        let mut score: i64 = 0;
        if let Some(pref) = self.preferred_vendor {
            if info.vendor == pref {
                score += PREFERRED_VENDOR_BONUS;
            }
        }
        #[expect(
            clippy::cast_possible_wrap,
            reason = "VRAM in GiB always fits in i64 (max ~16 EiB)"
        )]
        {
            score += (info.vram_bytes / (1024 * 1024 * 1024)) as i64;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "GFLOPS/100 always fits in i64 range"
        )]
        {
            score += (info.estimated_gflops / GFLOPS_DIVISOR) as i64;
        }
        if info.is_discrete {
            score += DISCRETE_BONUS;
        }
        if !info.is_busy() {
            score += IDLE_BONUS;
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
    ///
    /// `Arc<str>` instead of `String` so that cloning `DeviceInfo` (which
    /// happens on every device lease) is a ref-count bump, not a heap alloc.
    pub name: Arc<str>,
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
    #[expect(
        clippy::cast_precision_loss,
        reason = "u64→f64 for ratio; sub-ULP precision is irrelevant for a percentage"
    )]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device_info(
        vendor: GpuVendor,
        driver: GpuDriver,
        vram_gb: u64,
        gflops: f64,
        discrete: bool,
    ) -> DeviceInfo {
        DeviceInfo {
            index: 0,
            pool_index: 0,
            name: Arc::from("Test GPU"),
            vendor,
            driver,
            vram_bytes: vram_gb * 1024 * 1024 * 1024,
            estimated_gflops: gflops,
            is_discrete: discrete,
            allocations: Arc::new(AtomicUsize::new(0)),
            allocated_bytes: Arc::new(AtomicU64::new(0)),
            busy: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn workload_config_default() {
        let cfg = WorkloadConfig::default();
        assert_eq!(cfg.max_parallel, 4);
        assert!(cfg.prefer_discrete);
        assert!(cfg.exclude_software);
        assert!((cfg.min_gflops - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn device_requirements_builder() {
        let req = DeviceRequirements::new()
            .with_min_vram_gb(8)
            .prefer_nvidia()
            .require_discrete()
            .with_min_gflops(500.0);
        assert_eq!(req.min_vram_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(req.preferred_vendor, Some(GpuVendor::Nvidia));
        assert!(req.require_discrete);
        assert_eq!(req.min_gflops, Some(500.0));
        assert!(req.exclude_software);
    }

    #[test]
    fn score_excludes_software_when_required() {
        let req = DeviceRequirements::new();
        let info = make_device_info(GpuVendor::Software, GpuDriver::Software, 4, 50.0, false);
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_excludes_integrated_when_discrete_required() {
        let req = DeviceRequirements::new().require_discrete();
        let info = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            false,
        );
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_excludes_below_min_vram() {
        let req = DeviceRequirements::new().with_min_vram_gb(16);
        let info = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            true,
        );
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_excludes_below_min_gflops() {
        let req = DeviceRequirements::new().with_min_gflops(2000.0);
        let info = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            24,
            1000.0,
            true,
        );
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_prefers_vendor_bonus() {
        let req = DeviceRequirements::new().prefer_nvidia();
        let nvidia = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            true,
        );
        let amd = make_device_info(GpuVendor::Amd, GpuDriver::Radv, 8, 1000.0, true);
        let s_nvidia = req.score(&nvidia).unwrap();
        let s_amd = req.score(&amd).unwrap();
        assert!(
            s_nvidia > s_amd,
            "NVIDIA ({s_nvidia}) should score higher than AMD ({s_amd})"
        );
    }

    #[test]
    fn score_prefers_more_vram() {
        let req = DeviceRequirements {
            exclude_software: false,
            ..DeviceRequirements::default()
        };
        let big = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            24,
            1000.0,
            true,
        );
        let small = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            true,
        );
        assert!(req.score(&big).unwrap() > req.score(&small).unwrap());
    }

    #[test]
    fn score_discrete_bonus() {
        let req = DeviceRequirements {
            exclude_software: false,
            ..DeviceRequirements::default()
        };
        let discrete = make_device_info(GpuVendor::Intel, GpuDriver::Intel, 8, 500.0, true);
        let integrated = make_device_info(GpuVendor::Intel, GpuDriver::Intel, 8, 500.0, false);
        assert!(req.score(&discrete).unwrap() > req.score(&integrated).unwrap());
    }

    #[test]
    fn score_idle_bonus() {
        let req = DeviceRequirements {
            exclude_software: false,
            ..DeviceRequirements::default()
        };
        let idle = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            true,
        );
        let busy = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            true,
        );
        busy.busy.store(true, Ordering::Relaxed);
        assert!(req.score(&idle).unwrap() > req.score(&busy).unwrap());
    }

    #[test]
    fn device_info_available_vram() {
        let info = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            true,
        );
        let one_gb = 1024 * 1024 * 1024_u64;
        info.allocated_bytes.store(one_gb, Ordering::Relaxed);
        assert_eq!(info.available_vram_bytes(), 7 * one_gb);
    }

    #[test]
    fn device_info_usage_percent() {
        let info = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            10,
            1000.0,
            true,
        );
        let half = 5 * 1024 * 1024 * 1024_u64;
        info.allocated_bytes.store(half, Ordering::Relaxed);
        assert!((info.usage_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn device_info_zero_vram_usage() {
        let mut info = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            0,
            0.0,
            false,
        );
        info.vram_bytes = 0;
        assert!((info.usage_percent()).abs() < f64::EPSILON);
    }

    #[test]
    fn f64_builtins_supported_by_proprietary_not_nvk() {
        let prop = make_device_info(
            GpuVendor::Nvidia,
            GpuDriver::NvidiaProprietary,
            8,
            1000.0,
            true,
        );
        let nvk = make_device_info(GpuVendor::Nvidia, GpuDriver::Nvk, 8, 1000.0, true);
        let sw = make_device_info(GpuVendor::Software, GpuDriver::Software, 4, 50.0, false);
        assert!(prop.supports_f64_builtins());
        assert!(!nvk.supports_f64_builtins());
        assert!(!sw.supports_f64_builtins());
    }
}
