// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn test_elementwise_add() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input_a[idx] + input_b[idx];
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[1.0, 2.0, 3.0, 4.0]));
    bindings.insert(
        (0, 1),
        SimBuffer::from_f32_readonly(&[10.0, 20.0, 30.0, 40.0]),
    );
    bindings.insert((0, 2), SimBuffer::from_f32(&[0.0; 4]));

    exec.dispatch((4, 1, 1), &mut bindings).unwrap();

    let result = bindings[&(0, 2)].as_f32();
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_elementwise_mul() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    b[gid.x] = a[gid.x] * 2.0;
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[1.0, 2.0, 3.0]));
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 3]));

    exec.dispatch((3, 1, 1), &mut bindings).unwrap();
    assert_eq!(bindings[&(0, 1)].as_f32(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_math_builtins() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = input[gid.x];
    output[gid.x] = sin(x);
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert(
        (0, 0),
        SimBuffer::from_f32_readonly(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]),
    );
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 3]));

    exec.dispatch((3, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 1)].as_f32();
    assert!(result[0].abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!(result[2].abs() < 1e-6);
}

#[test]
fn test_f64_native() {
    let wgsl = "enable f16;\n\n\
@group(0) @binding(0) var<storage, read> input: array<f32>;\n\
@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n\n\
@compute @workgroup_size(1)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    output[gid.x] = input[gid.x] * input[gid.x];\n\
}\n";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[3.0, 7.0]));
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 2]));

    exec.dispatch((2, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 1)].as_f32();
    assert!((result[0] - 9.0).abs() < 1e-6);
    assert!((result[1] - 49.0).abs() < 1e-6);
}

#[test]
fn test_conditional_clamp() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = input[gid.x];
    output[gid.x] = max(min(x, 1.0), 0.0);
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[-0.5, 0.3, 0.7, 1.5]));
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

    exec.dispatch((4, 1, 1), &mut bindings).unwrap();
    assert_eq!(bindings[&(0, 1)].as_f32(), vec![0.0, 0.3, 0.7, 1.0]);
}

#[test]
fn test_relu() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = input[gid.x];
    output[gid.x] = max(x, 0.0);
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert(
        (0, 0),
        SimBuffer::from_f32_readonly(&[-2.0, -0.5, 0.0, 0.5, 2.0]),
    );
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 5]));

    exec.dispatch((5, 1, 1), &mut bindings).unwrap();
    assert_eq!(bindings[&(0, 1)].as_f32(), vec![0.0, 0.0, 0.0, 0.5, 2.0]);
}

#[test]
fn test_workgroup_size_larger_than_one() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] + 1.0;
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert(
        (0, 0),
        SimBuffer::from_f32_readonly(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    );
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 8]));

    exec.dispatch((2, 1, 1), &mut bindings).unwrap();
    assert_eq!(
        bindings[&(0, 1)].as_f32(),
        vec![11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0]
    );
}

#[test]
fn test_exp_log_roundtrip() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = log(exp(input[gid.x]));
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[0.0, 1.0, 2.0, -1.0]));
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

    exec.dispatch((4, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 1)].as_f32();
    for (i, &expected) in [0.0f32, 1.0, 2.0, -1.0].iter().enumerate() {
        assert!(
            (result[i] - expected).abs() < 1e-5,
            "index {i}: got {}, expected {expected}",
            result[i]
        );
    }
}

#[test]
fn test_tanh_activation() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = tanh(input[gid.x]);
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert(
        (0, 0),
        SimBuffer::from_f32_readonly(&[0.0, 1.0, -1.0, 100.0]),
    );
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

    exec.dispatch((4, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 1)].as_f32();
    assert!(result[0].abs() < 1e-6);
    assert!((result[1] - 0.761_594_2).abs() < 1e-5);
    assert!((result[2] - (-0.761_594_2)).abs() < 1e-5);
    assert!((result[3] - 1.0).abs() < 1e-6);
}

// ── f64 native tests ─────────────────────────────────────────────

#[test]
fn test_f64_elementwise_add() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = a[gid.x] + b[gid.x];
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert(
        (0, 0),
        SimBuffer::from_f64(&[1.000_000_000_000_1, 2.000_000_000_000_2]),
    );
    bindings.insert(
        (0, 1),
        SimBuffer::from_f64(&[0.000_000_000_000_1, 0.000_000_000_000_2]),
    );
    bindings.insert((0, 2), SimBuffer::from_f64(&[0.0; 2]));

    exec.dispatch((2, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 2)].as_f64();
    assert!(
        (result[0] - 1.000_000_000_000_2).abs() < 1e-13,
        "f64 precision lost: {}",
        result[0]
    );
    assert!(
        (result[1] - 2.000_000_000_000_4).abs() < 1e-13,
        "f64 precision lost: {}",
        result[1]
    );
}

#[test]
fn test_f64_transcendentals() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = sin(input[gid.x]);
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert(
        (0, 0),
        SimBuffer::from_f64(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]),
    );
    bindings.insert((0, 1), SimBuffer::from_f64(&[0.0; 3]));

    exec.dispatch((3, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 1)].as_f64();
    assert!(result[0].abs() < 1e-15, "sin(0) = {}", result[0]);
    assert!((result[1] - 1.0).abs() < 1e-15, "sin(pi/2) = {}", result[1]);
    assert!(result[2].abs() < 1e-15, "sin(pi) = {}", result[2]);
}

#[test]
#[expect(
    clippy::needless_type_cast,
    reason = "f64 is intentional: test verifies f32 loses 1e-15 precision"
)]
fn test_f64_precision_vs_f32() {
    let val: f64 = 1.0 + 1e-15;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "deliberate: testing that f32 cannot represent 1e-15 precision"
    )]
    let f32_val = val as f32;
    assert!(
        (f32_val - 1.0_f32).abs() < f32::EPSILON,
        "f32 should lose the 1e-15 precision"
    );

    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] * input[gid.x];
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f64(&[1.000_000_000_000_01]));
    bindings.insert((0, 1), SimBuffer::from_f64(&[0.0]));

    exec.dispatch((1, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 1)].as_f64();
    let expected = 1.000_000_000_000_01_f64 * 1.000_000_000_000_01_f64;
    assert!(
        (result[0] - expected).abs() < 1e-14,
        "f64 precision: got {}, expected {expected}",
        result[0]
    );
}

#[test]
fn test_f64_exp_log_roundtrip() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = log(exp(input[gid.x]));
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f64(&[0.0, 1.0, 2.0, -1.0, 0.5]));
    bindings.insert((0, 1), SimBuffer::from_f64(&[0.0; 5]));

    exec.dispatch((5, 1, 1), &mut bindings).unwrap();
    let result = bindings[&(0, 1)].as_f64();
    for (i, &expected) in [0.0, 1.0, 2.0, -1.0, 0.5].iter().enumerate() {
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "index {i}: got {}, expected {expected} (f64 precision)",
            result[i]
        );
    }
}

// ── Workgroup shared memory + barrier tests ──────────────────────

#[test]
fn test_shared_memory_reverse() {
    let wgsl = r"
var<workgroup> wg_data: array<f32, 4>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    wg_data[lid.x] = input[gid.x];
    workgroupBarrier();
    output[gid.x] = wg_data[3u - lid.x];
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[1.0, 2.0, 3.0, 4.0]));
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

    exec.dispatch((1, 1, 1), &mut bindings).unwrap();
    assert_eq!(bindings[&(0, 1)].as_f32(), vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_shared_memory_broadcast() {
    let wgsl = r"
var<workgroup> leader_val: f32;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    if lid.x == 0u {
        leader_val = input[0u];
    }
    workgroupBarrier();
    output[gid.x] = leader_val;
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[42.0, 0.0, 0.0, 0.0]));
    bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

    exec.dispatch((1, 1, 1), &mut bindings).unwrap();
    assert_eq!(bindings[&(0, 1)].as_f32(), vec![42.0, 42.0, 42.0, 42.0]);
}

// ── Atomic operation tests ───────────────────────────────────────

#[test]
fn test_atomic_add_accumulate() {
    let wgsl = r"
var<workgroup> sum: atomic<u32>;

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    atomicAdd(&sum, input[gid.x]);
    workgroupBarrier();
    if lid.x == 0u {
        output[0u] = atomicLoad(&sum);
    }
}
";
    let exec = NagaExecutor::new(wgsl, "main").unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert((0, 0), SimBuffer::from_u32(&[1, 2, 3, 4]));
    bindings.insert((0, 1), SimBuffer::from_u32(&[0]));

    exec.dispatch((1, 1, 1), &mut bindings).unwrap();
    assert_eq!(bindings[&(0, 1)].as_u32(), vec![10]);
}
