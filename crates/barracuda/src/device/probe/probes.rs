// SPDX-License-Identifier: AGPL-3.0-or-later
//! Probe shader definitions — one per f64 built-in function
//!
//! Each probe is compiled in an isolated shader so a crash in one does not
//! mask detection of others.

// Tolerances vary by operation type
const PROBE_F64_TOLERANCE_TIGHT: f64 = 1e-14;
const PROBE_F64_TOLERANCE_STANDARD: f64 = 1e-10;
const PROBE_F64_TOLERANCE_RELAXED: f64 = 1e-6;

/// One probe shader per function. Each must be compiled and dispatched
/// independently so a crash in one does not suppress detection of others.
pub(super) struct ProbeShader {
    pub name: &'static str,
    pub wgsl: &'static str,
    /// Expected result written to out[0]
    pub expected: f64,
    /// Acceptable absolute error
    pub tolerance: f64,
}

// ── DF64 probe shader source fragments ──────────────────────────────────────
// Self-contained WGSL shaders that test DF64 arithmetic and workgroup patterns.
// Each includes the minimal DF64 core inline (probes must be independent).

const PROBE_DF64_ARITH_WGSL: &str = "\
struct Df64 { hi: f32, lo: f32, }\n\
fn two_sum(a: f32, b: f32) -> Df64 {\n\
    let s = a + b;\n\
    let v = s - a;\n\
    let e = (a - (s - v)) + (b - v);\n\
    return Df64(s, e);\n\
}\n\
fn df64_add(a: Df64, b: Df64) -> Df64 {\n\
    let s = two_sum(a.hi, b.hi);\n\
    let e = a.lo + b.lo;\n\
    let v = two_sum(s.hi, s.lo + e);\n\
    return v;\n\
}\n\
fn df64_to_f64(v: Df64) -> f64 {\n\
    return f64(v.hi) + f64(v.lo);\n\
}\n\
@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
@compute @workgroup_size(1)\n\
fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
    let one = Df64(1.0, 0.0);\n\
    let two = df64_add(one, one);\n\
    out[0] = df64_to_f64(two);\n\
}";

const PROBE_DF64_FMA_TWO_PROD_WGSL: &str = "\
struct Df64 { hi: f32, lo: f32, }\n\
fn two_prod(a: f32, b: f32) -> Df64 {\n\
    let p = a * b;\n\
    let e = fma(a, b, -p);\n\
    return Df64(p, e);\n\
}\n\
@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
@compute @workgroup_size(1)\n\
fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
    let x: f32 = 1234567.0;\n\
    let y: f32 = 7654321.0;\n\
    let t = two_prod(x, y);\n\
    out[0] = f64(t.hi) + f64(t.lo);\n\
}";

const PROBE_DF64_WORKGROUP_REDUCE_WGSL: &str = "\
struct Df64 { hi: f32, lo: f32, }\n\
fn two_sum(a: f32, b: f32) -> Df64 {\n\
    let s = a + b;\n\
    let v = s - a;\n\
    let e = (a - (s - v)) + (b - v);\n\
    return Df64(s, e);\n\
}\n\
fn df64_add(a: Df64, b: Df64) -> Df64 {\n\
    let s = two_sum(a.hi, b.hi);\n\
    let e = a.lo + b.lo;\n\
    let v = two_sum(s.hi, s.lo + e);\n\
    return v;\n\
}\n\
fn df64_to_f64(v: Df64) -> f64 {\n\
    return f64(v.hi) + f64(v.lo);\n\
}\n\
@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
var<workgroup> shared_hi: array<f32, 256>;\n\
var<workgroup> shared_lo: array<f32, 256>;\n\
@compute @workgroup_size(256)\n\
fn probe(@builtin(local_invocation_id) lid: vec3<u32>) {\n\
    let tid = lid.x;\n\
    shared_hi[tid] = 1.0;\n\
    shared_lo[tid] = 0.0;\n\
    workgroupBarrier();\n\
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {\n\
        if (tid < stride) {\n\
            let a = Df64(shared_hi[tid], shared_lo[tid]);\n\
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);\n\
            let s = df64_add(a, b);\n\
            shared_hi[tid] = s.hi;\n\
            shared_lo[tid] = s.lo;\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
    if (tid == 0u) {\n\
        out[0] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));\n\
    }\n\
}";

pub(super) const PROBES: &[ProbeShader] = &[
    ProbeShader {
        name: "basic_f64",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   let x: f64 = f64(3.0);\n\
                   let y: f64 = x * f64(2.0) + f64(1.0);\n\
                   out[0] = y;\n\
               }",
        expected: 7.0,
        tolerance: PROBE_F64_TOLERANCE_TIGHT,
    },
    ProbeShader {
        name: "exp",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = exp(f64(1.0));\n\
               }",
        expected: std::f64::consts::E,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    ProbeShader {
        name: "log",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = log(f64(2.718281828459045));\n\
               }",
        expected: 1.0,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    ProbeShader {
        name: "exp2",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = exp2(f64(3.0));\n\
               }",
        expected: 8.0,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
    ProbeShader {
        name: "log2",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = log2(f64(8.0));\n\
               }",
        expected: 3.0,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
    ProbeShader {
        name: "sin",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = sin(f64(9.21460183660255));\n\
               }",
        expected: 0.156_434_465_040_230_9,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    ProbeShader {
        name: "cos",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = cos(f64(9.21460183660255));\n\
               }",
        expected: -0.987_688_340_595_137_8,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    ProbeShader {
        name: "sqrt",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = sqrt(f64(2.0));\n\
               }",
        expected: std::f64::consts::SQRT_2,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
    ProbeShader {
        name: "fma",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = fma(f64(2.0), f64(3.0), f64(1.0));\n\
               }",
        expected: 7.0,
        tolerance: PROBE_F64_TOLERANCE_TIGHT,
    },
    ProbeShader {
        name: "abs_min_max",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   let a = abs(f64(-3.5));\n\
                   let b = min(a, f64(4.0));\n\
                   let c = max(b, f64(2.0));\n\
                   out[0] = c;\n\
               }",
        expected: 3.5,
        tolerance: PROBE_F64_TOLERANCE_TIGHT,
    },
    ProbeShader {
        name: "shared_mem_f64",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               var<workgroup> wg_data: array<f64, 4>;\n\
               @compute @workgroup_size(4)\n\
               fn probe(@builtin(local_invocation_id) lid: vec3<u32>) {\n\
                   wg_data[lid.x] = f64(lid.x + 1u);\n\
                   workgroupBarrier();\n\
                   if lid.x == 0u {\n\
                       out[0] = wg_data[0] + wg_data[1] + wg_data[2] + wg_data[3];\n\
                   }\n\
               }",
        expected: 10.0,
        tolerance: PROBE_F64_TOLERANCE_TIGHT,
    },
    // ── Composite transcendental probe ───────────────────────────────────────
    // NVVM can compile individual f64 transcendentals but crashes when combining
    // log+exp+sqrt in a single shader. This probe catches that failure mode.
    ProbeShader {
        name: "composite_transcendental",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   let x = f64(2.0);\n\
                   let a = log(x);\n\
                   let b = exp(a);\n\
                   let c = sqrt(b);\n\
                   let d = sin(c) * sin(c) + cos(c) * cos(c);\n\
                   out[0] = d;\n\
               }",
        expected: 1.0,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    // Chained lgamma-like pattern: exp(log(Gamma)) where the shader exercises
    // the same op mix as Bessel K₀ / Beta function shaders.
    ProbeShader {
        name: "exp_log_chain",
        wgsl: "@group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   let x = f64(3.0);\n\
                   let lg = log(exp(x));\n\
                   let r = exp(lg - log(x));\n\
                   out[0] = r * sqrt(abs(f64(1.0)));\n\
               }",
        expected: 1.0,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    // ── DF64 (f32-pair) capability probes ────────────────────────────────────
    // These test whether DF64 arithmetic patterns compile and execute correctly.
    // The production ReduceScalarPipeline depends on these patterns.
    ProbeShader {
        name: "df64_arith",
        wgsl: PROBE_DF64_ARITH_WGSL,
        expected: 2.0,
        tolerance: PROBE_F64_TOLERANCE_TIGHT,
    },
    ProbeShader {
        name: "df64_fma_two_prod",
        wgsl: PROBE_DF64_FMA_TWO_PROD_WGSL,
        // 1234567.0_f32 and 7654321.0_f32 are exact in f32.
        // two_prod reconstructs the exact f64 product: 1234567 * 7654321
        expected: 9_449_772_114_007.0,
        tolerance: 1024.0,
    },
    ProbeShader {
        name: "df64_workgroup_reduce",
        wgsl: PROBE_DF64_WORKGROUP_REDUCE_WGSL,
        expected: 256.0,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
];
