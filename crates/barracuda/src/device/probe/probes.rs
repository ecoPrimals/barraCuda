// SPDX-License-Identifier: AGPL-3.0-only
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

pub(super) const PROBES: &[ProbeShader] = &[
    ProbeShader {
        name: "basic_f64",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
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
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = exp(f64(1.0));\n\
               }",
        expected: std::f64::consts::E,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    ProbeShader {
        name: "log",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = log(f64(2.718281828459045));\n\
               }",
        expected: 1.0,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    ProbeShader {
        name: "exp2",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = exp2(f64(3.0));\n\
               }",
        expected: 8.0,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
    ProbeShader {
        name: "log2",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = log2(f64(8.0));\n\
               }",
        expected: 3.0,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
    ProbeShader {
        name: "sin",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = sin(f64(1.5707963267948966));\n\
               }",
        expected: 1.0,
        tolerance: PROBE_F64_TOLERANCE_RELAXED,
    },
    ProbeShader {
        name: "cos",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = cos(f64(0.0));\n\
               }",
        expected: 1.0,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
    ProbeShader {
        name: "sqrt",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = sqrt(f64(2.0));\n\
               }",
        expected: std::f64::consts::SQRT_2,
        tolerance: PROBE_F64_TOLERANCE_STANDARD,
    },
    ProbeShader {
        name: "fma",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               @compute @workgroup_size(1)\n\
               fn probe(@builtin(global_invocation_id) _id: vec3<u32>) {\n\
                   out[0] = fma(f64(2.0), f64(3.0), f64(1.0));\n\
               }",
        expected: 7.0,
        tolerance: PROBE_F64_TOLERANCE_TIGHT,
    },
    ProbeShader {
        name: "abs_min_max",
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
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
        wgsl: "enable f64;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
               var<workgroup> shared: array<f64, 4>;\n\
               @compute @workgroup_size(4)\n\
               fn probe(@builtin(local_invocation_id) lid: vec3<u32>) {\n\
                   shared[lid.x] = f64(lid.x + 1u);\n\
                   workgroupBarrier();\n\
                   if lid.x == 0u {\n\
                       out[0] = shared[0] + shared[1] + shared[2] + shared[3];\n\
                   }\n\
               }",
        expected: 10.0,
        tolerance: PROBE_F64_TOLERANCE_TIGHT,
    },
];
