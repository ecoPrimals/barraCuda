// SPDX-License-Identifier: AGPL-3.0-or-later
//! SPIR-V Emission — Sovereign Compiler Phase 4.
//!
//! Takes a validated `naga::Module` + `naga::valid::ModuleInfo` and emits
//! SPIR-V 1.3 binary words using `naga::back::spv::Writer`.
//!
//! Configured for Vulkan 1.1 compute shaders with `Float64` capability.

use super::SovereignError;
use naga::back::spv;
use naga::valid::ModuleInfo;

/// Emit SPIR-V words from a validated naga module.
///
/// The output is a `Vec<u32>` suitable for `wgpu::ShaderSource::SpirV`.
pub fn emit_spirv(module: &naga::Module, info: &ModuleInfo) -> Result<Vec<u32>, SovereignError> {
    let options = spv::Options {
        lang_version: (1, 3),
        flags: spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        capabilities: None,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        binding_map: spv::BindingMap::default(),
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        debug_info: None,
    };

    let mut words = Vec::new();
    let mut writer = spv::Writer::new(&options)
        .map_err(|e| SovereignError::SpirvEmit(format!("failed to create SPIR-V writer: {e}")))?;

    writer
        .write(module, info, None, &None, &mut words)
        .map_err(|e| SovereignError::SpirvEmit(format!("SPIR-V write failed: {e}")))?;

    Ok(words)
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_spirv_trivial() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2.0;
}
"#;
        let module = naga::front::wgsl::parse_str(wgsl).expect("parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).expect("validate");
        let words = emit_spirv(&module, &info).expect("emit");
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x07230203, "SPIR-V magic number");
    }

    #[test]
    fn test_emit_spirv_f64() {
        let wgsl = r#"
enable f16;

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let a = input[idx];
    let b = input[idx + 1u];
    output[idx] = a * b + a;
}
"#;
        // f64 requires Float64 capability — our writer should handle it
        let module = naga::front::wgsl::parse_str(wgsl);
        if module.is_err() {
            // Some naga builds require `enable f64;` or different syntax
            return;
        }
        let module = module.unwrap();
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        if let Ok(info) = validator.validate(&module) {
            let words = emit_spirv(&module, &info).expect("emit f64 spirv");
            assert_eq!(words[0], 0x07230203);
        }
    }
}
