// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL Re-emission — safe alternative to SPIR-V passthrough.
//!
//! Takes a validated, optimised `naga::Module` + `naga::valid::ModuleInfo` and
//! emits WGSL text via `naga::back::wgsl::Writer`. The resulting WGSL can be
//! fed back through `wgpu::Device::create_shader_module` (the safe API),
//! preserving all naga-level optimisations (FMA fusion, dead expression
//! elimination) without requiring `unsafe` SPIR-V passthrough.

use super::SovereignError;
use naga::valid::ModuleInfo;

/// Emit optimised WGSL from a validated naga module.
///
/// # Errors
///
/// Returns [`SovereignError::WgslEmit`] if the naga WGSL backend writer
/// fails (e.g. unsupported expression after optimisation).
pub fn emit_wgsl(module: &naga::Module, info: &ModuleInfo) -> Result<String, SovereignError> {
    let flags = naga::back::wgsl::WriterFlags::empty();
    naga::back::wgsl::write_string(module, info, flags)
        .map_err(|e| SovereignError::WgslEmit(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_wgsl_roundtrip() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2.0;
}
";
        let module = naga::front::wgsl::parse_str(wgsl).expect("parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).expect("validate");
        let emitted = emit_wgsl(&module, &info).expect("emit");
        assert!(
            emitted.contains("output"),
            "emitted WGSL should reference output"
        );
        assert!(
            emitted.contains("input"),
            "emitted WGSL should reference input"
        );
    }

    #[test]
    fn test_emit_wgsl_preserves_fma() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read> a_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_buf: array<f32>;
@group(0) @binding(2) var<storage, read> c_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = fma(a_buf[i], b_buf[i], c_buf[i]);
}
";
        let module = naga::front::wgsl::parse_str(wgsl).expect("parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).expect("validate");
        let emitted = emit_wgsl(&module, &info).expect("emit");
        assert!(emitted.contains("fma"), "FMA should survive WGSL roundtrip");
    }
}
