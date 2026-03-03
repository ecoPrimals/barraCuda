// SPDX-License-Identifier: AGPL-3.0-or-later
//! Core device type — minimal enum for routing and capability modules.
//!
//! Extracted from unified.rs to avoid circular dependencies. The full `Device`
//! API (info, select_for_workload, etc.) lives in `unified.rs`.

use std::fmt;

/// Unified device abstraction — hardware type only.
///
/// Full construction and capability logic lives in `unified` and `capabilities`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU execution (pure Rust)
    CPU,

    /// GPU execution (WGSL via wgpu)
    GPU,

    /// NPU execution (Akida neuromorphic)
    NPU,

    /// TPU execution (Tensor Processing Unit)
    TPU,

    /// Automatic selection based on workload
    Auto,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            Device::GPU => write!(f, "GPU"),
            Device::NPU => write!(f, "NPU"),
            Device::TPU => write!(f, "TPU"),
            Device::Auto => write!(f, "Auto"),
        }
    }
}
