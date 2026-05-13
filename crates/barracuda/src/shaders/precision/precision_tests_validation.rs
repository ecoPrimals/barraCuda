// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision validation tests — edge cases, E2E, and fault recovery.

use super::*;

#[test]
fn test_downcast_df64_only_maps_existing_transcendentals() {
    let source = "let a = exp_f64(x); let b = tan_f64(y); let c = sqrt_f64(z);";
    let result = downcast_f64_to_df64(source);
    assert!(result.contains("exp_df64("), "exp should map");
    assert!(result.contains("sqrt_df64("), "sqrt should map");
    assert!(
        result.contains("tan_f64("),
        "tan_f64 should stay unmapped (no df64 impl)"
    );
}

#[test]
fn test_downcast_f32_mixed_u32_f64() {
    let source = "let n: u32 = 100u;\nlet x: f64 = f64(1.5);\nlet y: f64 = input[n];";
    let result = downcast_f64_to_f32(source);
    assert!(result.contains("n: u32"), "u32 type preserved");
    assert!(result.contains("x: f32"), "f64 downcasts to f32");
    assert!(result.contains("f32(1.5)"), "constructor downcasts");
}

#[test]
fn test_clamp_f64_range_handles_all_patterns() {
    let patterns = vec![
        ("-1.7976931348623157e+308", "-3.4028235e+38"),
        ("1.7976931348623157e+308", "3.4028235e+38"),
        ("-1e308", "-3.4028235e+38"),
        ("1e308", "3.4028235e+38"),
        ("-1e300", "-3.4028235e+38"),
        ("1e300", "3.4028235e+38"),
    ];
    for (input, expected) in &patterns {
        let result = downcast_f64_to_f32(input);
        assert!(
            result.contains(expected),
            "pattern {input} should become {expected}, got {result}"
        );
    }
}

/// Universal shader with comparison ops validates at all 3 tiers.
const UNIVERSAL_COMPARISON: &str = r"
@group(0) @binding(0) var<storage, read> a: array<Scalar>;
@group(0) @binding(1) var<storage, read> b: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    if (op_gt(a[idx], b[idx])) {
        output[idx] = a[idx];
    } else {
        output[idx] = b[idx];
    }
}
";

#[cfg(feature = "gpu")]
#[test]
fn test_e2e_comparison_shader_all_precisions() {
    for prec in [Precision::F32, Precision::F64] {
        let preamble = prec.op_preamble();
        let source = format!("{preamble}\n{UNIVERSAL_COMPARISON}");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{prec:?} comparison parse: {e}"));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("{prec:?} comparison validate: {e}"));
    }
    const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../shaders/math/df64_transcendentals.wgsl");
    let preamble = Precision::Df64.op_preamble();
    let source = format!("{DF64_CORE}\n{DF64_TRANS}\n{preamble}\n{UNIVERSAL_COMPARISON}");
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("DF64 comparison parse: {e}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("DF64 comparison validate: {e}"));
}

/// Universal shader with pack/unpack validates at DF64.
const UNIVERSAL_PACK_UNPACK: &str = r"
@group(0) @binding(0) var<storage, read> input: array<StorageType>;
@group(0) @binding(1) var<storage, read_write> output: array<StorageType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    let val = op_unpack(input[idx]);
    let result = op_add(val, op_one());
    output[idx] = op_pack(result);
}
";

#[cfg(feature = "gpu")]
#[test]
fn test_e2e_pack_unpack_df64() {
    const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../shaders/math/df64_transcendentals.wgsl");
    let preamble = Precision::Df64.op_preamble();
    let source = format!("{DF64_CORE}\n{DF64_TRANS}\n{preamble}\n{UNIVERSAL_PACK_UNPACK}");
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("DF64 pack/unpack parse: {e}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("DF64 pack/unpack validate: {e}"));
}

/// E2E: f64 canonical shader with transcendentals downcasts to f32 correctly.
#[test]
fn test_e2e_transcendental_downcast_f32() {
    let f64_shader = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

fn activate(x: f64) -> f64 {
    return tanh_f64(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = activate(input[idx]);
}
";
    let f32_source = downcast_f64_to_f32_with_transcendentals(f64_shader);
    assert!(f32_source.contains("array<f32>"), "storage downcast");
    assert!(f32_source.contains("tanh("), "tanh_f64 → tanh");
    assert!(!f32_source.contains("f64"), "no f64 should remain");
    let module = naga::front::wgsl::parse_str(&f32_source)
        .unwrap_or_else(|e| panic!("f32 downcast parse: {e}\n{f32_source}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("f32 downcast validate: {e}"));
}

#[test]
fn test_fault_preamble_consistency_under_concatenation() {
    for prec in [
        Precision::Binary,
        Precision::Int2,
        Precision::Q4,
        Precision::Q8,
        Precision::Fp8E5M2,
        Precision::Fp8E4M3,
        Precision::Bf16,
        Precision::F32,
        Precision::F64,
        Precision::Df64,
    ] {
        let p = prec.op_preamble();
        let opens = p.matches('{').count();
        let closes = p.matches('}').count();
        assert_eq!(opens, closes, "{prec:?} preamble has unbalanced braces");
    }
}

#[test]
fn test_fault_downcast_idempotent_f32() {
    let source = "let x: f64 = f64(1.0);";
    let once = downcast_f64_to_f32(source);
    let twice = downcast_f64_to_f32(&once);
    assert_eq!(once, twice, "double downcast should be idempotent");
}

#[test]
fn test_fault_precision_bytes_consistent() {
    assert_eq!(Precision::Binary.bytes_per_element(), 1);
    assert_eq!(Precision::Fp8E4M3.bytes_per_element(), 1);
    assert_eq!(Precision::Bf16.bytes_per_element(), 2);
    assert_eq!(Precision::F16.bytes_per_element(), 2);
    assert_eq!(Precision::F32.bytes_per_element(), 4);
    assert_eq!(Precision::F64.bytes_per_element(), 8);
    assert_eq!(Precision::Df64.bytes_per_element(), 8);
    assert_eq!(Precision::Qf128.bytes_per_element(), 16);
    assert_eq!(Precision::Df128.bytes_per_element(), 16);
}

#[test]
fn test_fault_precision_is_f64_class() {
    assert!(!Precision::F32.is_f64_class());
    assert!(Precision::F64.is_f64_class());
    assert!(Precision::Df64.is_f64_class());
    assert!(Precision::Df128.is_f64_class());
    assert!(!Precision::Qf128.is_f64_class());
}

#[test]
fn test_precision_reduced_and_extended() {
    assert!(Precision::Binary.is_reduced());
    assert!(Precision::Bf16.is_reduced());
    assert!(Precision::F16.is_reduced());
    assert!(!Precision::F32.is_reduced());
    assert!(Precision::Qf128.is_extended());
    assert!(Precision::Df128.is_extended());
    assert!(!Precision::F64.is_extended());
}

#[test]
fn test_precision_is_quantized() {
    assert!(Precision::Binary.is_quantized());
    assert!(Precision::Int2.is_quantized());
    assert!(Precision::Q4.is_quantized());
    assert!(Precision::Q8.is_quantized());
    assert!(!Precision::Fp8E4M3.is_quantized());
    assert!(!Precision::F32.is_quantized());
}

// ══════════════════════════════════════════════════════════════════════════════
// DF64 NVK E2E — GPU dispatch tests for the production compile_shader_df64 path
// ══════════════════════════════════════════════════════════════════════════════
//
// These tests close the gap between naga-only validation (above) and real GPU
// execution. They exercise WgpuDevice::compile_shader_df64 → ComputeDispatch
// → readback → numerical verification against CPU reference values.
//
// Gate: requires real GPU with SHADER_F64 feature (for f64 buffer types).
// DF64 computation is entirely in f32 pairs, so broken native f64 arithmetic
// (e.g. NVK/NAK) is fine — that is exactly the DF64 use case.

/// DF64 add+mul kernel: out[i] = df64(a[i]) * df64(b[i]) + df64(c[i])
/// Tests the full production compilation path including df64_core prepend.
const DF64_E2E_FMA_KERNEL: &str = r"
@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read> c: array<f64>;
@group(0) @binding(3) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&out) { return; }
    let da = df64_from_f64(a[i]);
    let db = df64_from_f64(b[i]);
    let dc = df64_from_f64(c[i]);
    let product = df64_mul(da, db);
    let result = df64_add(product, dc);
    out[i] = df64_to_f64(result);
}
";

/// DF64 Kahan-style summation kernel: reduces N values to a single sum.
/// Exercises df64_add in a loop — the pattern used by production reducers.
const DF64_E2E_KAHAN_KERNEL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> out: array<f64>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _gid: vec3<u32>) {
    var acc = df64_from_f64(f64(0.0));
    for (var i = 0u; i < n; i++) {
        acc = df64_add(acc, df64_from_f64(input[i]));
    }
    out[0] = df64_to_f64(acc);
}
";

#[tokio::test]
async fn test_df64_e2e_fma_gpu_dispatch() {
    use crate::device::compute_pipeline::ComputeDispatch;
    use crate::device::test_pool::get_test_device_if_gpu_available;
    use wgpu::util::DeviceExt;

    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    if !device.has_f64_shaders() {
        return;
    }

    let n = 256usize;
    let a_data: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.1, 1.0)).collect();
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.05, 0.5)).collect();
    let c_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();

    let expected: Vec<f64> = (0..n)
        .map(|i| a_data[i].mul_add(b_data[i], c_data[i]))
        .collect();

    let a_buf = device
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("a"),
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let b_buf = device
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("b"),
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let c_buf = device
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c"),
            contents: bytemuck::cast_slice(&c_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_buf = device
        .device()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

    ComputeDispatch::new(&*device, "df64_e2e_fma")
        .shader(DF64_E2E_FMA_KERNEL, "main")
        .df64()
        .storage_read(0, &a_buf)
        .storage_read(1, &b_buf)
        .storage_read(2, &c_buf)
        .storage_rw(3, &out_buf)
        .dispatch_1d(n as u32)
        .submit()
        .expect("DF64 FMA dispatch should succeed");

    let results = device
        .read_f64_buffer(&out_buf, n)
        .expect("readback should succeed");

    for (i, (&got, &exp)) in results.iter().zip(expected.iter()).enumerate() {
        let rel_err = if exp.abs() > 1e-15 {
            (got - exp).abs() / exp.abs()
        } else {
            (got - exp).abs()
        };
        assert!(
            rel_err < 1e-6,
            "DF64 FMA E2E mismatch at [{i}]: got {got}, expected {exp}, rel_err {rel_err}"
        );
    }
}

#[tokio::test]
async fn test_df64_e2e_kahan_summation_gpu_dispatch() {
    use crate::device::compute_pipeline::ComputeDispatch;
    use crate::device::test_pool::get_test_device_if_gpu_available;
    use wgpu::util::DeviceExt;

    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    if !device.has_f64_shaders() {
        return;
    }

    let n = 1000u32;
    let input_data: Vec<f64> = (0..n).map(|i| 1.0 / ((i as f64) + 1.0)).collect();
    let expected: f64 = input_data.iter().sum();

    let input_buf = device
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_buf = device
        .device()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    let n_buf = device
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("n"),
            contents: bytemuck::cast_slice(&[n, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    ComputeDispatch::new(&*device, "df64_e2e_kahan")
        .shader(DF64_E2E_KAHAN_KERNEL, "main")
        .df64()
        .storage_read(0, &input_buf)
        .storage_rw(1, &out_buf)
        .uniform(2, &n_buf)
        .dispatch(1, 1, 1)
        .submit()
        .expect("DF64 Kahan dispatch should succeed");

    let results = device
        .read_f64_buffer(&out_buf, 1)
        .expect("readback should succeed");

    let rel_err = (results[0] - expected).abs() / expected.abs();
    // DF64 ≈ 48-bit mantissa; serial chain of 1000 additions degrades to ~1e-5
    assert!(
        rel_err < 1e-5,
        "DF64 Kahan E2E: got {}, expected {expected}, rel_err {rel_err}",
        results[0]
    );
}
