// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign compute backend via coralReef's `coral-gpu` crate.
//!
//! `CoralReefDevice` implements `GpuBackend` by dispatching directly through
//! `coral-gpu::GpuContext`, bypassing the entire wgpu/Vulkan/Metal stack.
//! This is the bridge between barraCuda (Layer 1) and the sovereign pipeline
//! (Layers 2-4).
//!
//! # Status
//!
//! **Scaffold** — the struct and trait implementation are in place. Method
//! bodies will be filled when `coral-gpu` is published as a standalone
//! `cargo add` dependency. Until then, all methods return
//! `BarracudaError::Device` with a clear message.
//!
//! # Activation
//!
//! ```toml
//! [features]
//! sovereign-dispatch = ["gpu"]
//! ```
//!
//! # Architecture
//!
//! ```text
//! barraCuda op
//!   → ComputeDispatch<CoralReefDevice>::submit()
//!     → CoralReefDevice::dispatch_compute()
//!       → coral_gpu::GpuContext::compile_wgsl() → native binary
//!       → coral_gpu::GpuContext::alloc() + upload()
//!       → coral_gpu::GpuContext::dispatch()
//!       → coral_gpu::GpuContext::sync() + readback()
//! ```

use super::backend::{DispatchDescriptor, GpuBackend};
use crate::error::{BarracudaError, Result};

/// Buffer handle for the sovereign compute path.
///
/// Wraps a `coral-gpu` buffer allocation. Currently a placeholder that
/// tracks size for the scaffold implementation. All fields are plain data
/// types, making this automatically `Send + Sync`.
#[derive(Debug, Clone, Copy)]
pub struct CoralBuffer {
    _id: u64,
    _size: u64,
}

/// Sovereign GPU compute device via coralReef.
///
/// When `coral-gpu` is available, this wraps `coral_gpu::GpuContext` and
/// provides the full compute surface: buffer allocation, WGSL compilation
/// to native GPU binaries, and direct dispatch without Vulkan.
///
/// First target: AMD RDNA2 (GFX1030) — E2E verified in coralReef Phase 10.
/// NVIDIA: pending hardware validation (SM70 codegen exists, dispatch untested).
#[derive(Debug)]
pub struct CoralReefDevice {
    name: String,
    // When coral-gpu is available:
    // context: coral_gpu::GpuContext,
}

impl CoralReefDevice {
    /// Create a new `CoralReefDevice`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] until `coral-gpu` is available as a dependency.
    /// When available, returns [`Err`] if no compatible GPU is found or
    /// if the coral-gpu context cannot be initialized.
    pub fn new() -> Result<Self> {
        Err(BarracudaError::Device(
            "CoralReefDevice: coral-gpu crate not yet available as dependency — \
             sovereign dispatch is scaffolded but not yet functional. \
             Track progress in SOVEREIGN_PIPELINE_TRACKER.md"
                .into(),
        ))
    }

    fn scaffold_err<T>(method: &str) -> Result<T> {
        Err(BarracudaError::Device(format!(
            "CoralReefDevice::{method}: coral-gpu not yet available"
        )))
    }
}

impl GpuBackend for CoralReefDevice {
    type Buffer = CoralBuffer;

    fn name(&self) -> &str {
        &self.name
    }

    fn has_f64_shaders(&self) -> bool {
        false
    }

    fn is_lost(&self) -> bool {
        false
    }

    fn alloc_buffer(&self, _label: &str, _size: u64) -> Result<CoralBuffer> {
        Self::scaffold_err("alloc_buffer")
    }

    fn alloc_buffer_init(&self, _label: &str, _contents: &[u8]) -> Result<CoralBuffer> {
        Self::scaffold_err("alloc_buffer_init")
    }

    fn alloc_uniform(&self, _label: &str, _contents: &[u8]) -> Result<CoralBuffer> {
        Self::scaffold_err("alloc_uniform")
    }

    fn upload(&self, _buffer: &CoralBuffer, _offset: u64, _data: &[u8]) {}

    fn download(&self, _buffer: &CoralBuffer, _size: u64) -> Result<Vec<u8>> {
        Self::scaffold_err("download")
    }

    fn dispatch_compute(&self, _desc: DispatchDescriptor<'_, Self>) -> Result<()> {
        Self::scaffold_err("dispatch_compute")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_returns_clear_error() {
        let err = CoralReefDevice::new().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("coral-gpu"),
            "error should mention coral-gpu: {msg}"
        );
        assert!(
            msg.contains("SOVEREIGN_PIPELINE_TRACKER"),
            "error should point to tracker doc: {msg}"
        );
    }
}
