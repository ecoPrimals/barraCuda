// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL shader template constants for precision-generic code generation

/// Element-wise addition: C = A + B
pub const TEMPLATE_ELEMENTWISE_ADD: &str = r"// Element-wise Addition: C = A + B
// Generated for precision: {{SCALAR}}

@group(0) @binding(0) var<storage, read> a: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> b: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    
    // Core algorithm (identical to CPU implementation)
    output[idx] = a[idx] + b[idx];
}
{{#if HAS_VEC4}}

// Vectorized variant for better memory throughput
struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }

@group(0) @binding(0) var<storage, read> a_vec: array<{{VEC4}}>;
@group(0) @binding(1) var<storage, read> b_vec: array<{{VEC4}}>;
@group(0) @binding(2) var<storage, read_write> out_vec: array<{{VEC4}}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main_vec4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size / 4u) { out_vec[idx] = a_vec[idx] + b_vec[idx]; }
}
{{/if}}
";

/// Element-wise multiplication: C = A * B
pub const TEMPLATE_ELEMENTWISE_MUL: &str = r"// Element-wise Multiplication: C = A * B
// Generated for precision: {{SCALAR}}

@group(0) @binding(0) var<storage, read> a: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> b: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    
    // Core algorithm
    output[idx] = a[idx] * b[idx];
}
{{#if HAS_VEC4}}

struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }

@group(0) @binding(0) var<storage, read> a_vec: array<{{VEC4}}>;
@group(0) @binding(1) var<storage, read> b_vec: array<{{VEC4}}>;
@group(0) @binding(2) var<storage, read_write> out_vec: array<{{VEC4}}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main_vec4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size / 4u) { out_vec[idx] = a_vec[idx] * b_vec[idx]; }
}
{{/if}}
";

/// Fused Multiply-Add: D = A * B + C
pub const TEMPLATE_ELEMENTWISE_FMA: &str = r"// Fused Multiply-Add: D = A * B + C
// Generated for precision: {{SCALAR}}

@group(0) @binding(0) var<storage, read> a: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> b: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read> c: array<{{SCALAR}}>;
@group(0) @binding(3) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    
    // FMA - single rounding, better precision than separate mul+add
    output[idx] = fma(a[idx], b[idx], c[idx]);
}
";

/// Dot product: sum(A * B)
pub const TEMPLATE_DOT_PRODUCT: &str = r"// Dot Product: sum(A * B)
// Generated for precision: {{SCALAR}}
// Uses workgroup reduction for parallel summation

var<workgroup> shared: array<{{SCALAR}}, 256>;

@group(0) @binding(0) var<storage, read> a: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> b: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;
    let n = arrayLength(&a);
    
    // Load and multiply
    if (gid < n) {
        shared[lid] = a[gid] * b[gid];
    } else {
        shared[lid] = {{SCALAR}}(0);
    }
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared[lid] = shared[lid] + shared[lid + stride];
        }
        workgroupBarrier();
    }
    
    // Write partial sum
    if (lid == 0u) {
        output[workgroup_id.x] = shared[0];
    }
}
";

/// Reduction sum: sum(A)
pub const TEMPLATE_REDUCE_SUM: &str = r"// Reduction Sum: sum(A)
// Generated for precision: {{SCALAR}}

var<workgroup> shared: array<{{SCALAR}}, 256>;

@group(0) @binding(0) var<storage, read> input: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;
    let n = arrayLength(&input);
    
    shared[lid] = select({{SCALAR}}(0), input[gid], gid < n);
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared[lid] = shared[lid] + shared[lid + stride];
        }
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        output[workgroup_id.x] = shared[0];
    }
}
";

/// Element-wise subtraction: C = A - B
pub const TEMPLATE_ELEMENTWISE_SUB: &str = r"// Element-wise Subtraction: C = A - B
// Generated for precision: {{SCALAR}}

@group(0) @binding(0) var<storage, read> a: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> b: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = a[idx] - b[idx];
}
";

/// Element-wise absolute value: B = |A|
pub const TEMPLATE_ELEMENTWISE_ABS: &str = r"// Element-wise Absolute Value: B = |A|
// Generated for precision: {{SCALAR}}

@group(0) @binding(0) var<storage, read> input: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = abs(input[idx]);
}
";

/// Element-wise negation: B = -A
pub const TEMPLATE_ELEMENTWISE_NEG: &str = r"// Element-wise Negation: B = -A
// Generated for precision: {{SCALAR}}

@group(0) @binding(0) var<storage, read> input: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read_write> output: array<{{SCALAR}}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = -input[idx];
}
";

/// Element-wise clamp: B = clamp(A, lo, hi)
pub const TEMPLATE_ELEMENTWISE_CLAMP: &str = r"// Element-wise Clamp: B = clamp(A, lo, hi)
// Generated for precision: {{SCALAR}}

struct ClampParams { lo: {{SCALAR}}, hi: {{SCALAR}}, }

@group(0) @binding(0) var<storage, read> input: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read_write> output: array<{{SCALAR}}>;
@group(0) @binding(2) var<uniform> params: ClampParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = clamp(input[idx], params.lo, params.hi);
}
";

/// Mean reduction: mean(A)
pub const TEMPLATE_REDUCE_MEAN: &str = r"// Reduction Mean: mean(A)
// Generated for precision: {{SCALAR}}
// Two-pass: first pass produces partial sums + counts, second pass divides

var<workgroup> shared: array<{{SCALAR}}, 256>;

struct MeanParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read_write> output: array<{{SCALAR}}>;
@group(0) @binding(2) var<uniform> params: MeanParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;

    shared[lid] = select({{SCALAR}}(0), input[gid], gid < params.size);
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared[lid] = shared[lid] + shared[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        output[workgroup_id.x] = shared[0] / {{SCALAR}}(params.size);
    }
}
";

/// MSE loss: mean((pred - target)^2)
pub const TEMPLATE_MSE_LOSS: &str = r"// Mean Squared Error Loss
// Generated for precision: {{SCALAR}}

var<workgroup> shared: array<{{SCALAR}}, 256>;

struct LossParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> pred: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> target: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{SCALAR}}>;
@group(0) @binding(3) var<uniform> params: LossParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;

    if (gid < params.size) {
        let diff = pred[gid] - target[gid];
        shared[lid] = diff * diff;
    } else {
        shared[lid] = {{SCALAR}}(0);
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared[lid] = shared[lid] + shared[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        output[workgroup_id.x] = shared[0];
    }
}
";

/// MAE loss: mean(|pred - target|)
pub const TEMPLATE_MAE_LOSS: &str = r"// Mean Absolute Error Loss
// Generated for precision: {{SCALAR}}

var<workgroup> shared: array<{{SCALAR}}, 256>;

struct LossParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> pred: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> target: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{SCALAR}}>;
@group(0) @binding(3) var<uniform> params: LossParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;

    if (gid < params.size) {
        shared[lid] = abs(pred[gid] - target[gid]);
    } else {
        shared[lid] = {{SCALAR}}(0);
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared[lid] = shared[lid] + shared[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        output[workgroup_id.x] = shared[0];
    }
}
";

/// Scaled addition (SAXPY): C = alpha * A + B
pub const TEMPLATE_SAXPY: &str = r"// Scaled Addition: C = alpha * A + B
// Generated for precision: {{SCALAR}}

struct SaxpyParams { alpha: {{SCALAR}}, }

@group(0) @binding(0) var<storage, read> a: array<{{SCALAR}}>;
@group(0) @binding(1) var<storage, read> b: array<{{SCALAR}}>;
@group(0) @binding(2) var<storage, read_write> output: array<{{SCALAR}}>;
@group(0) @binding(3) var<uniform> params: SaxpyParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = params.alpha * a[idx] + b[idx];
}
";

/// Remove a conditional block from the template
pub fn remove_conditional_block(source: &str, start_marker: &str, end_marker: &str) -> String {
    let mut result = String::new();
    let mut in_block = false;

    for line in source.lines() {
        if line.contains(start_marker) {
            in_block = true;
            continue;
        }
        if line.contains(end_marker) {
            in_block = false;
            continue;
        }
        if !in_block {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}
