// SPDX-License-Identifier: AGPL-3.0-or-later
//! # barracuda-spirv: SPIR-V Passthrough Bridge
//!
//! Isolates the single `unsafe` call to `wgpu::Device::create_shader_module_passthrough`
//! so the main `barracuda` crate can maintain `#![forbid(unsafe_code)]`.
//!
//! ## Safety Contract
//!
//! The only `unsafe` in this crate is the call to `create_shader_module_passthrough`.
//! The caller (`barracuda`'s sovereign compiler) guarantees SPIR-V provenance through
//! the `ValidatedSpirv` type, which can only be constructed after naga validation.
//! By the time words reach this bridge, they have been:
//!
//! 1. Parsed from valid WGSL by naga's frontend
//! 2. Validated by `naga::valid::Validator`
//! 3. Emitted by naga's SPIR-V backend from that validated IR
//!
//! The bridge accepts `&[u32]` to avoid a circular dependency with `barracuda`.
//! The type-level safety enforcement remains in `barracuda` via `ValidatedSpirv`.
//!
//! ## Evolution Path
//!
//! When wgpu exposes a safe `create_shader_module_trusted` API (tracking
//! wgpu#4854), this crate collapses to a safe wrapper and the `unsafe`
//! disappears entirely. Until then, this is the minimal, auditable surface.

#![warn(missing_docs)]
#![deny(unsafe_code)]

use std::borrow::Cow;
use std::fmt;

/// Errors from SPIR-V passthrough compilation.
#[derive(Debug)]
pub enum SpirvError {
    /// Caller passed an empty SPIR-V module (never valid per the spec).
    EmptyModule,
}

impl fmt::Display for SpirvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModule => {
                write!(
                    f,
                    "SPIR-V words must be non-empty — empty modules are never valid"
                )
            }
        }
    }
}

impl std::error::Error for SpirvError {}

/// Compile SPIR-V words into a wgpu shader module via the passthrough path.
///
/// This bypasses naga's WGSL re-emission entirely, feeding SPIR-V directly to
/// the Vulkan/Metal/DX12 backend.
///
/// # Errors
///
/// Returns [`SpirvError::EmptyModule`] if `spirv_words` is empty.
///
/// # Safety Argument (not `unsafe fn`)
///
/// This function is safe to call because the unsafe operation inside is
/// bounded by a contract enforced at the type level in the `barracuda` crate:
/// only `ValidatedSpirv::words()` can produce the `&[u32]` that reaches here,
/// and `ValidatedSpirv` can only be constructed from naga-validated IR.
///
/// The `#[expect(unsafe_code)]` below is the sole exception to the crate-level
/// `#![deny(unsafe_code)]`, auditable in one place.
pub fn compile_spirv_passthrough(
    device: &wgpu::Device,
    spirv_words: &[u32],
    label: Option<&str>,
) -> Result<wgpu::ShaderModule, SpirvError> {
    if spirv_words.is_empty() {
        return Err(SpirvError::EmptyModule);
    }

    let desc = wgpu::ShaderModuleDescriptorPassthrough {
        label,
        spirv: Some(Cow::Borrowed(spirv_words)),
        entry_point: String::new(),
        num_workgroups: (0, 0, 0),
        runtime_checks: wgpu::ShaderRuntimeChecks::unchecked(),
        dxil: None,
        msl: None,
        hlsl: None,
        glsl: None,
        wgsl: None,
    };

    // SAFETY: The SPIR-V was produced by naga's SPIR-V backend from a
    // naga-validated IR module. The caller (barracuda's sovereign compiler)
    // enforces this through the ValidatedSpirv type — only code paths that
    // pass naga::valid::Validator::validate() can construct ValidatedSpirv,
    // and only ValidatedSpirv::words() can produce the &[u32] passed here.
    #[expect(
        unsafe_code,
        reason = "wgpu passthrough requires unsafe until wgpu#4854 lands"
    )]
    Ok(unsafe { device.create_shader_module_passthrough(desc) })
}
