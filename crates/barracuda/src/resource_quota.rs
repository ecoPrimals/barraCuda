// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resource Quota Management for GPU Compute
//!
//! Provides VRAM budget enforcement and resource tracking for multi-tenant
//! GPU workloads. Enables fair resource sharing across tasks and prevents
//! memory exhaustion.
//!
//! # Architecture
//!
//! ```text
//! ResourceQuota (per-task budget)
//!     ↓
//! QuotaTracker (enforces limits, tracks usage)
//!     ↓
//! DeviceQuota (per-device allocation + usage)
//!     ↓
//! QuotaPool (multi-device quota-aware pool)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::resource_quota::{ResourceQuota, QuotaPool};
//!
//! // Create a quota: 2GB VRAM, prefer NVIDIA
//! let quota = ResourceQuota::new()
//!     .with_max_vram_bytes(2 * 1024 * 1024 * 1024)
//!     .with_preferred_vendor(GpuVendor::Nvidia);
//!
//! // Get a device from the pool that fits the quota
//! let pool = QuotaPool::new().await?;
//! let device = pool.acquire_device(&quota)?;
//!
//! // Track allocations against quota
//! let tracker = QuotaTracker::new(quota);
//! tracker.try_allocate(1024 * 1024)?;  // 1 MB - succeeds
//! tracker.try_allocate(3 * 1024 * 1024 * 1024)?;  // 3 GB - fails (exceeds quota)
//! ```
//!
//! # Deep Debt Compliance
//!
//! - Modern idiomatic Rust (parameter-based APIs, no global state mutation)
//! - Zero unsafe code
//! - Capability-based (devices discovered at runtime)
//! - Proper error handling (Result types, no panics)

use crate::error::{BarracudaError, Result};
use crate::multi_gpu::GpuVendor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// ResourceQuota - Budget Definition
// ============================================================================

/// Resource quota defining limits for a task/tenant
///
/// Quotas are immutable once created. Use the builder pattern to configure.
#[derive(Debug, Clone)]
pub struct ResourceQuota {
    /// Maximum VRAM bytes allowed (None = unlimited)
    pub max_vram_bytes: Option<u64>,

    /// Maximum number of buffers allowed (None = unlimited)
    pub max_buffers: Option<usize>,

    /// Maximum single buffer size (None = unlimited)
    pub max_single_buffer_bytes: Option<u64>,

    /// Preferred GPU vendor (None = no preference)
    pub preferred_vendor: Option<GpuVendor>,

    /// Minimum VRAM required to be usable
    pub min_vram_bytes: Option<u64>,

    /// Quota identifier for logging/debugging
    pub name: String,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_vram_bytes: None,
            max_buffers: None,
            max_single_buffer_bytes: None,
            preferred_vendor: None,
            min_vram_bytes: None,
            name: "default".to_string(),
        }
    }
}

impl ResourceQuota {
    /// Create a new quota with default (unlimited) settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a quota with a specific name
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Set maximum VRAM bytes
    #[must_use]
    pub fn with_max_vram_bytes(mut self, bytes: u64) -> Self {
        self.max_vram_bytes = Some(bytes);
        self
    }

    /// Set maximum VRAM in megabytes (convenience)
    #[must_use]
    pub fn with_max_vram_mb(self, mb: u64) -> Self {
        self.with_max_vram_bytes(mb * 1024 * 1024)
    }

    /// Set maximum VRAM in gigabytes (convenience)
    #[must_use]
    pub fn with_max_vram_gb(self, gb: u64) -> Self {
        self.with_max_vram_bytes(gb * 1024 * 1024 * 1024)
    }

    /// Set maximum number of buffers
    #[must_use]
    pub fn with_max_buffers(mut self, count: usize) -> Self {
        self.max_buffers = Some(count);
        self
    }

    /// Set maximum single buffer size
    #[must_use]
    pub fn with_max_single_buffer_bytes(mut self, bytes: u64) -> Self {
        self.max_single_buffer_bytes = Some(bytes);
        self
    }

    /// Set preferred GPU vendor
    #[must_use]
    pub fn with_preferred_vendor(mut self, vendor: GpuVendor) -> Self {
        self.preferred_vendor = Some(vendor);
        self
    }

    /// Set minimum VRAM required
    #[must_use]
    pub fn with_min_vram_bytes(mut self, bytes: u64) -> Self {
        self.min_vram_bytes = Some(bytes);
        self
    }

    /// Set minimum VRAM in gigabytes (convenience)
    #[must_use]
    pub fn with_min_vram_gb(self, gb: u64) -> Self {
        self.with_min_vram_bytes(gb * 1024 * 1024 * 1024)
    }

    /// Check if a device meets the minimum requirements
    pub fn device_meets_requirements(&self, device_vram_bytes: u64, vendor: GpuVendor) -> bool {
        // Check minimum VRAM
        if let Some(min) = self.min_vram_bytes {
            if device_vram_bytes < min {
                return false;
            }
        }

        // Vendor preference is soft (doesn't disqualify)
        // But if we have a preference and it doesn't match, log it
        if let Some(pref) = self.preferred_vendor {
            if pref != vendor {
                tracing::debug!(
                    "Quota '{}': device vendor {:?} doesn't match preference {:?}",
                    self.name,
                    vendor,
                    pref
                );
            }
        }

        true
    }
}

// ============================================================================
// QuotaTracker - Usage Tracking and Enforcement
// ============================================================================

/// Tracks resource usage against a quota
///
/// Thread-safe via atomic operations. Create one per task/session.
#[derive(Debug)]
pub struct QuotaTracker {
    /// The quota being enforced
    quota: ResourceQuota,

    /// Current VRAM usage in bytes
    current_vram_bytes: AtomicU64,

    /// Current buffer count
    current_buffers: AtomicU64,

    /// Peak VRAM usage (for diagnostics)
    peak_vram_bytes: AtomicU64,

    /// Total bytes allocated (including freed)
    total_allocated_bytes: AtomicU64,

    /// Number of allocation failures due to quota
    quota_failures: AtomicU64,
}

impl QuotaTracker {
    /// Create a new tracker for the given quota
    #[must_use]
    pub fn new(quota: ResourceQuota) -> Self {
        Self {
            quota,
            current_vram_bytes: AtomicU64::new(0),
            current_buffers: AtomicU64::new(0),
            peak_vram_bytes: AtomicU64::new(0),
            total_allocated_bytes: AtomicU64::new(0),
            quota_failures: AtomicU64::new(0),
        }
    }

    /// Create a tracker wrapped in Arc for sharing
    #[must_use]
    pub fn new_shared(quota: ResourceQuota) -> Arc<Self> {
        Arc::new(Self::new(quota))
    }

    /// Get the quota being tracked
    pub fn quota(&self) -> &ResourceQuota {
        &self.quota
    }

    /// Current VRAM usage in bytes
    pub fn current_vram_bytes(&self) -> u64 {
        self.current_vram_bytes.load(Ordering::Relaxed)
    }

    /// Current buffer count
    pub fn current_buffers(&self) -> u64 {
        self.current_buffers.load(Ordering::Relaxed)
    }

    /// Peak VRAM usage in bytes
    pub fn peak_vram_bytes(&self) -> u64 {
        self.peak_vram_bytes.load(Ordering::Relaxed)
    }

    /// Total bytes allocated since creation
    pub fn total_allocated_bytes(&self) -> u64 {
        self.total_allocated_bytes.load(Ordering::Relaxed)
    }

    /// Number of quota failures
    pub fn quota_failures(&self) -> u64 {
        self.quota_failures.load(Ordering::Relaxed)
    }

    /// Remaining VRAM budget (None if unlimited)
    pub fn remaining_vram_bytes(&self) -> Option<u64> {
        self.quota.max_vram_bytes.map(|max| {
            let current = self.current_vram_bytes();
            max.saturating_sub(current)
        })
    }

    /// Usage as percentage of quota (0-100, or None if unlimited)
    pub fn usage_percent(&self) -> Option<f64> {
        self.quota.max_vram_bytes.map(|max| {
            let current = self.current_vram_bytes();
            (current as f64 / max as f64) * 100.0
        })
    }

    /// Check if an allocation would exceed quota (without actually allocating)
    pub fn would_exceed_quota(&self, bytes: u64) -> bool {
        // Check single buffer limit
        if let Some(max_single) = self.quota.max_single_buffer_bytes {
            if bytes > max_single {
                return true;
            }
        }

        // Check total VRAM limit
        if let Some(max_vram) = self.quota.max_vram_bytes {
            let current = self.current_vram_bytes();
            if current + bytes > max_vram {
                return true;
            }
        }

        // Check buffer count limit
        if let Some(max_buffers) = self.quota.max_buffers {
            let current = self.current_buffers() as usize;
            if current >= max_buffers {
                return true;
            }
        }

        false
    }

    /// Try to allocate bytes, respecting quota
    ///
    /// Returns Ok(()) if allocation is allowed, Err if it would exceed quota.
    /// On success, the allocation is tracked. Call `deallocate` when done.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn try_allocate(&self, bytes: u64) -> Result<()> {
        // Check single buffer limit
        if let Some(max_single) = self.quota.max_single_buffer_bytes {
            if bytes > max_single {
                self.quota_failures.fetch_add(1, Ordering::Relaxed);
                return Err(BarracudaError::resource_exhausted(format!(
                    "Quota '{}': single buffer {} bytes exceeds limit {} bytes",
                    self.quota.name, bytes, max_single
                )));
            }
        }

        // Check buffer count limit
        if let Some(max_buffers) = self.quota.max_buffers {
            let current = self.current_buffers() as usize;
            if current >= max_buffers {
                self.quota_failures.fetch_add(1, Ordering::Relaxed);
                return Err(BarracudaError::resource_exhausted(format!(
                    "Quota '{}': buffer count {} at limit {}",
                    self.quota.name, current, max_buffers
                )));
            }
        }

        // Atomically try to add to VRAM usage
        if let Some(max_vram) = self.quota.max_vram_bytes {
            loop {
                let current = self.current_vram_bytes.load(Ordering::Relaxed);
                let new_total = current + bytes;

                if new_total > max_vram {
                    self.quota_failures.fetch_add(1, Ordering::Relaxed);
                    return Err(BarracudaError::resource_exhausted(format!(
                        "Quota '{}': allocation of {} bytes would exceed limit ({} + {} > {})",
                        self.quota.name, bytes, current, bytes, max_vram
                    )));
                }

                // Try to atomically update
                if self
                    .current_vram_bytes
                    .compare_exchange(current, new_total, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
                // If CAS failed, loop and retry
            }
        } else {
            // No VRAM limit, just track
            self.current_vram_bytes.fetch_add(bytes, Ordering::Relaxed);
        }

        // Update buffer count
        self.current_buffers.fetch_add(1, Ordering::Relaxed);

        // Update stats
        self.total_allocated_bytes
            .fetch_add(bytes, Ordering::Relaxed);

        // Update peak
        let current = self.current_vram_bytes();
        let mut peak = self.peak_vram_bytes.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_vram_bytes.compare_exchange(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }

        Ok(())
    }

    /// Record deallocation of bytes
    pub fn deallocate(&self, bytes: u64) {
        self.current_vram_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.current_buffers.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get a summary string for logging/debugging
    pub fn summary(&self) -> String {
        let current = self.current_vram_bytes();
        let peak = self.peak_vram_bytes();
        let buffers = self.current_buffers();
        let failures = self.quota_failures();

        self.quota.max_vram_bytes.map_or(
            format!(
                "Quota '{}': {} bytes used (unlimited), {} buffers, peak {} bytes",
                self.quota.name,
                format_bytes(current),
                buffers,
                format_bytes(peak)
            ),
            |max| {
                let percent = (current as f64 / max as f64) * 100.0;
                format!(
                    "Quota '{}': {:.1}% used ({} / {} bytes), {} buffers, peak {} bytes, {} failures",
                    self.quota.name,
                    percent,
                    format_bytes(current),
                    format_bytes(max),
                    buffers,
                    format_bytes(peak),
                    failures
                )
            },
        )
    }

    /// Reset usage tracking (does not affect quota limits)
    pub fn reset(&self) {
        self.current_vram_bytes.store(0, Ordering::Relaxed);
        self.current_buffers.store(0, Ordering::Relaxed);
        self.peak_vram_bytes.store(0, Ordering::Relaxed);
        self.total_allocated_bytes.store(0, Ordering::Relaxed);
        self.quota_failures.store(0, Ordering::Relaxed);
    }
}

// ============================================================================
// DeviceQuota - Per-Device Allocation
// ============================================================================

/// Per-device quota information
#[derive(Debug, Clone)]
pub struct DeviceQuota {
    /// Device index in the pool
    pub device_index: usize,

    /// Device name
    pub device_name: String,

    /// Device vendor
    pub vendor: GpuVendor,

    /// Total VRAM on device (bytes)
    pub total_vram_bytes: u64,

    /// Currently allocated VRAM (bytes)
    pub allocated_vram_bytes: u64,

    /// Available VRAM (bytes)
    pub available_vram_bytes: u64,

    /// Number of active allocations
    pub active_allocations: usize,
}

impl DeviceQuota {
    /// Check if this device can accommodate an allocation
    #[must_use]
    pub fn can_allocate(&self, bytes: u64) -> bool {
        self.available_vram_bytes >= bytes
    }

    /// Get usage percentage (0-100)
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total_vram_bytes == 0 {
            return 0.0;
        }
        (self.allocated_vram_bytes as f64 / self.total_vram_bytes as f64) * 100.0
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

// ============================================================================
// Preset Quotas
// ============================================================================

/// Common quota presets
pub mod presets {
    use super::ResourceQuota;

    /// VRAM limit for small task preset (512 MiB).
    const PRESET_SMALL_VRAM_MB: u64 = 512;

    /// VRAM limit for medium task preset (2 GiB).
    const PRESET_MEDIUM_VRAM_GB: u64 = 2;

    /// VRAM limit for large task preset (8 GiB).
    const PRESET_LARGE_VRAM_GB: u64 = 8;

    /// VRAM limit for scientific preset (4 GiB).
    const PRESET_SCIENTIFIC_VRAM_GB: u64 = 4;

    /// Buffer limit for scientific preset.
    const PRESET_SCIENTIFIC_MAX_BUFFERS: usize = 1000;

    /// VRAM limit for ML inference preset (2 GiB).
    const PRESET_ML_INFERENCE_VRAM_GB: u64 = 2;

    /// Buffer limit for ML inference preset.
    const PRESET_ML_INFERENCE_MAX_BUFFERS: usize = 100;

    /// VRAM limit for ML training preset (8 GiB).
    const PRESET_ML_TRAINING_VRAM_GB: u64 = 8;

    /// Buffer limit for ML training preset.
    const PRESET_ML_TRAINING_MAX_BUFFERS: usize = 500;

    /// Small task: 512 MB VRAM
    #[must_use]
    pub fn small() -> ResourceQuota {
        ResourceQuota::named("small").with_max_vram_mb(PRESET_SMALL_VRAM_MB)
    }

    /// Medium task: 2 GB VRAM
    #[must_use]
    pub fn medium() -> ResourceQuota {
        ResourceQuota::named("medium").with_max_vram_gb(PRESET_MEDIUM_VRAM_GB)
    }

    /// Large task: 8 GB VRAM
    #[must_use]
    pub fn large() -> ResourceQuota {
        ResourceQuota::named("large").with_max_vram_gb(PRESET_LARGE_VRAM_GB)
    }

    /// Unlimited (for testing or privileged tasks)
    #[must_use]
    pub fn unlimited() -> ResourceQuota {
        ResourceQuota::named("unlimited")
    }

    /// Scientific computing: 4 GB VRAM, high buffer limits
    #[must_use]
    pub fn scientific() -> ResourceQuota {
        ResourceQuota::named("scientific")
            .with_max_vram_gb(PRESET_SCIENTIFIC_VRAM_GB)
            .with_max_buffers(PRESET_SCIENTIFIC_MAX_BUFFERS)
    }

    /// ML inference: 2 GB VRAM per model
    #[must_use]
    pub fn ml_inference() -> ResourceQuota {
        ResourceQuota::named("ml_inference")
            .with_max_vram_gb(PRESET_ML_INFERENCE_VRAM_GB)
            .with_max_buffers(PRESET_ML_INFERENCE_MAX_BUFFERS)
    }

    /// ML training: 8 GB VRAM, high buffer limits
    #[must_use]
    pub fn ml_training() -> ResourceQuota {
        ResourceQuota::named("ml_training")
            .with_max_vram_gb(PRESET_ML_TRAINING_VRAM_GB)
            .with_max_buffers(PRESET_ML_TRAINING_MAX_BUFFERS)
    }
}

#[cfg(test)]
#[path = "resource_quota_tests.rs"]
mod tests;
