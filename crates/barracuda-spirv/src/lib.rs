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

#![warn(missing_docs)]

use std::borrow::Cow;

/// Compile SPIR-V words into a wgpu shader module via the passthrough path.
///
/// This bypasses naga's WGSL re-emission entirely, feeding SPIR-V directly to
/// the Vulkan/Metal/DX12 backend.
///
/// # Safety
///
/// The caller must ensure `spirv_words` was produced by a trusted pipeline.
/// In `barracuda`, this is enforced by the `ValidatedSpirv` type which gates
/// construction to naga-validated IR only. The SPIR-V is passed directly to
/// the GPU driver without further validation by wgpu.
#[must_use]
pub fn compile_spirv_passthrough(
    device: &wgpu::Device,
    spirv_words: &[u32],
    label: Option<&str>,
) -> wgpu::ShaderModule {
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
    unsafe { device.create_shader_module_passthrough(desc) }
}
