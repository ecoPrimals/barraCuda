// SPDX-License-Identifier: AGPL-3.0-or-later
// ============================================================================
// fused_map_reduce_f64.wgsl — Single-dispatch map + reduce at f64 precision
// ============================================================================
//
// UNIFIED PATTERN (Feb 16 2026) — Serves all springs:
//   - wetSpring: Shannon entropy (-p * log(p)), Simpson index (p²)
//   - airSpring: ET₀ batched sums, water balance totals
//   - hotSpring: Convergence norms, energy functionals
//
// ARCHITECTURE:
//   Phase 1: Each thread maps multiple elements (grid-stride loop)
//   Phase 2: Workgroup tree reduction in shared memory
//   Phase 3: Thread 0 writes partial to global memory
//   Phase 4: (Optional) Second pass reduces partials to final scalar
//
// REQUIRES: SHADER_F64 feature
// PRECISION: Uses (zero + literal) pattern for full f64 constants
//
// Date: February 16, 2026
// License: AGPL-3.0-or-later
// ============================================================================

// ============================================================================
// log_f64 provided by math_f64.wgsl auto-injection

// ============================================================================
// REDUCE OPERATION ENUM (selected via params.reduce_op)
// ============================================================================
// 0 = SUM
// 1 = MAX
// 2 = MIN
// 3 = PRODUCT (log-domain for stability)

// ============================================================================
// MAP OPERATION ENUM (selected via params.map_op)
// ============================================================================
// 0 = IDENTITY (passthrough)
// 1 = SHANNON (-p * log(p) where p = x / total)
// 2 = SIMPSON (p² where p = x / total)
// 3 = SQUARE (x²)
// 4 = ABS (|x|)
// 5 = LOG (log(x))
// 6 = NEGATE (-x)

struct Params {
    n: u32,              // Input array length
    n_workgroups: u32,   // Number of dispatched workgroups (for grid stride)
    total: f64,          // Normalization constant (e.g., sum for Shannon)
    map_op: u32,         // Map operation enum
    reduce_op: u32,      // Reduce operation enum
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f64, 256>;

// ============================================================================
// MAP FUNCTION — Apply element-wise transformation
// ============================================================================
fn apply_map(x: f64, total: f64, map_op: u32) -> f64 {
    let zero = x - x;
    let one = zero + 1.0;
    let tiny = zero + 1e-300;
    
    switch (map_op) {
        case 0u: {
            // IDENTITY
            return x;
        }
        case 1u: {
            // SHANNON: -p * log(p) where p = x / total
            // Handle p = 0 case: 0 * log(0) → 0 (L'Hôpital)
            if (x <= zero) {
                return zero;
            }
            let p = x / total;
            if (p <= tiny) {
                return zero;
            }
            return -p * log_f64(p);
        }
        case 2u: {
            // SIMPSON: p² where p = x / total
            let p = x / total;
            return p * p;
        }
        case 3u: {
            // SQUARE: x²
            return x * x;
        }
        case 4u: {
            // ABS: |x|
            if (x < zero) {
                return -x;
            }
            return x;
        }
        case 5u: {
            // LOG: log(x)
            if (x <= zero) {
                let big = zero + 1e38;
                return -big * big;
            }
            return log_f64(x);
        }
        case 6u: {
            // NEGATE: -x
            return -x;
        }
        default: {
            return x;
        }
    }
}

// ============================================================================
// REDUCE FUNCTION — Combine two values
// ============================================================================
fn apply_reduce(a: f64, b: f64, reduce_op: u32) -> f64 {
    switch (reduce_op) {
        case 0u: {
            // SUM
            return a + b;
        }
        case 1u: {
            // MAX
            if (a > b) { return a; }
            return b;
        }
        case 2u: {
            // MIN
            if (a < b) { return a; }
            return b;
        }
        case 3u: {
            // PRODUCT (in log-domain: sum of logs)
            return a + b;
        }
        default: {
            return a + b;
        }
    }
}

// ============================================================================
// REDUCE IDENTITY — Starting value for reduction
// ============================================================================
fn reduce_identity(reduce_op: u32, sample: f64) -> f64 {
    let zero = sample - sample;
    let big = zero + 1e308;
    
    switch (reduce_op) {
        case 0u: {
            // SUM
            return zero;
        }
        case 1u: {
            // MAX
            return -big;
        }
        case 2u: {
            // MIN
            return big;
        }
        case 3u: {
            // PRODUCT (log-domain: sum starts at 0)
            return zero;
        }
        default: {
            return zero;
        }
    }
}

// ============================================================================
// MAIN ENTRY POINT — Fused map-reduce
// ============================================================================
@compute @workgroup_size(256)
fn fused_map_reduce(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = params.n;
    let total = params.total;
    let map_op = params.map_op;
    let reduce_op = params.reduce_op;
    
    // Phase 1: Grid-stride loop — each thread processes multiple elements
    let first_val = input[0];
    var acc = reduce_identity(reduce_op, first_val);
    
    // Grid stride = total threads = n_workgroups × workgroup_size (256)
    let grid_stride = params.n_workgroups * 256u;
    
    var idx = gid;
    while (idx < n) {
        let val = input[idx];
        let mapped = apply_map(val, total, map_op);
        acc = apply_reduce(acc, mapped, reduce_op);
        idx = idx + grid_stride;
    }
    
    // Phase 2: Store to shared memory
    shared_data[tid] = acc;
    workgroupBarrier();
    
    // Phase 3: Tree reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_data[tid] = apply_reduce(shared_data[tid], shared_data[tid + stride], reduce_op);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Phase 4: Thread 0 writes workgroup result to output
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// ============================================================================
// SECOND PASS — Reduce partial results to final scalar
// ============================================================================
@compute @workgroup_size(256)
fn reduce_partials(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let n = params.n;  // Number of partials
    let reduce_op = params.reduce_op;
    
    // Load partial or identity
    let first_val = input[0];
    var val: f64;
    if (tid < n) {
        val = input[tid];
    } else {
        val = reduce_identity(reduce_op, first_val);
    }
    
    shared_data[tid] = val;
    workgroupBarrier();
    
    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_data[tid] = apply_reduce(shared_data[tid], shared_data[tid + stride], reduce_op);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Thread 0 writes final result
    if (tid == 0u) {
        output[0] = shared_data[0];
    }
}
