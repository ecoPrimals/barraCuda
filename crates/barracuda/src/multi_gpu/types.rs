// SPDX-License-Identifier: AGPL-3.0-or-later
//! Types and configuration for multi-GPU workload distribution.

use super::topology::DeviceClass;
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
    /// Preferred device class.
    pub preferred_class: Option<DeviceClass>,
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

    /// Prefer discrete GPUs.
    #[must_use]
    pub fn prefer_discrete(mut self) -> Self {
        self.preferred_class = Some(DeviceClass::DiscreteGpu);
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
        if self.exclude_software && info.device_class == DeviceClass::Software {
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

        const PREFERRED_CLASS_BONUS: i64 = 1000;
        const DISCRETE_BONUS: i64 = 100;
        const IDLE_BONUS: i64 = 50;
        const GFLOPS_DIVISOR: f64 = 100.0;

        let mut score: i64 = 0;
        if let Some(pref) = self.preferred_class {
            if info.device_class == pref {
                score += PREFERRED_CLASS_BONUS;
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
    /// Device class (discrete / integrated / software / unknown).
    pub device_class: DeviceClass,
    /// Estimated VRAM in bytes.
    pub vram_bytes: u64,
    /// Estimated FP32 GFLOPS.
    pub estimated_gflops: f64,
    /// True if discrete (not integrated).
    pub is_discrete: bool,
    /// Whether native f64 builtins (exp, log, pow) work on this device.
    pub f64_builtins_available: bool,
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
        self.f64_builtins_available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device_info(
        device_class: DeviceClass,
        vram_gb: u64,
        gflops: f64,
        discrete: bool,
    ) -> DeviceInfo {
        DeviceInfo {
            index: 0,
            pool_index: 0,
            name: Arc::from("Test GPU"),
            device_class,
            vram_bytes: vram_gb * 1024 * 1024 * 1024,
            estimated_gflops: gflops,
            is_discrete: discrete,
            f64_builtins_available: !matches!(device_class, DeviceClass::Software),
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
            .prefer_discrete()
            .require_discrete()
            .with_min_gflops(500.0);
        assert_eq!(req.min_vram_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(req.preferred_class, Some(DeviceClass::DiscreteGpu));
        assert!(req.require_discrete);
        assert_eq!(req.min_gflops, Some(500.0));
        assert!(req.exclude_software);
    }

    #[test]
    fn score_excludes_software_when_required() {
        let req = DeviceRequirements::new();
        let info = make_device_info(DeviceClass::Software, 4, 50.0, false);
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_excludes_integrated_when_discrete_required() {
        let req = DeviceRequirements::new().require_discrete();
        let info = make_device_info(DeviceClass::IntegratedGpu, 8, 1000.0, false);
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_excludes_below_min_vram() {
        let req = DeviceRequirements::new().with_min_vram_gb(16);
        let info = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_excludes_below_min_gflops() {
        let req = DeviceRequirements::new().with_min_gflops(2000.0);
        let info = make_device_info(DeviceClass::DiscreteGpu, 24, 1000.0, true);
        assert!(req.score(&info).is_none());
    }

    #[test]
    fn score_prefers_class_bonus() {
        let req = DeviceRequirements::new().prefer_discrete();
        let discrete = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        let integrated = make_device_info(DeviceClass::IntegratedGpu, 8, 1000.0, false);
        let s_discrete = req.score(&discrete).unwrap();
        let s_integrated = req.score(&integrated).unwrap();
        assert!(
            s_discrete > s_integrated,
            "DiscreteGpu ({s_discrete}) should score higher than IntegratedGpu ({s_integrated})"
        );
    }

    #[test]
    fn score_prefers_more_vram() {
        let req = DeviceRequirements {
            exclude_software: false,
            ..DeviceRequirements::default()
        };
        let big = make_device_info(DeviceClass::DiscreteGpu, 24, 1000.0, true);
        let small = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        assert!(req.score(&big).unwrap() > req.score(&small).unwrap());
    }

    #[test]
    fn score_discrete_bonus() {
        let req = DeviceRequirements {
            exclude_software: false,
            ..DeviceRequirements::default()
        };
        let discrete = make_device_info(DeviceClass::DiscreteGpu, 8, 500.0, true);
        let integrated = make_device_info(DeviceClass::IntegratedGpu, 8, 500.0, false);
        assert!(req.score(&discrete).unwrap() > req.score(&integrated).unwrap());
    }

    #[test]
    fn score_idle_bonus() {
        let req = DeviceRequirements {
            exclude_software: false,
            ..DeviceRequirements::default()
        };
        let idle = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        let busy = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        busy.busy.store(true, Ordering::Relaxed);
        assert!(req.score(&idle).unwrap() > req.score(&busy).unwrap());
    }

    #[test]
    fn device_info_available_vram() {
        let info = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        let one_gb = 1024 * 1024 * 1024_u64;
        info.allocated_bytes.store(one_gb, Ordering::Relaxed);
        assert_eq!(info.available_vram_bytes(), 7 * one_gb);
    }

    #[test]
    fn device_info_usage_percent() {
        let info = make_device_info(DeviceClass::DiscreteGpu, 10, 1000.0, true);
        let half = 5 * 1024 * 1024 * 1024_u64;
        info.allocated_bytes.store(half, Ordering::Relaxed);
        assert!((info.usage_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn device_info_zero_vram_usage() {
        let mut info = make_device_info(DeviceClass::DiscreteGpu, 0, 0.0, false);
        info.vram_bytes = 0;
        assert!((info.usage_percent()).abs() < f64::EPSILON);
    }

    #[test]
    fn f64_builtins_based_on_field() {
        let discrete = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        assert!(discrete.f64_builtins_available);
        assert!(discrete.supports_f64_builtins());

        let integrated = make_device_info(DeviceClass::IntegratedGpu, 8, 1000.0, false);
        assert!(integrated.f64_builtins_available);
        assert!(integrated.supports_f64_builtins());

        let software = make_device_info(DeviceClass::Software, 4, 50.0, false);
        assert!(!software.f64_builtins_available);
        assert!(!software.supports_f64_builtins());

        let mut toggled = make_device_info(DeviceClass::DiscreteGpu, 8, 1000.0, true);
        toggled.f64_builtins_available = false;
        assert!(!toggled.supports_f64_builtins());
    }
}
