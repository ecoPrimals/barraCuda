// Index Add - Scatter-add operation with atomic CAS for overlapping indices (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)
//
// Uses atomic compare-and-swap on i32 bitcast to implement f32 atomic add.
// This correctly handles overlapping indices without race conditions.
// Note: Downcast to f32 for actual execution; f64 canonical for precision.

struct Params {
    size: u32,
    dim_size: u32,
    outer_size: u32,
    inner_size: u32,
    scatter_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<f64>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
// Atomic i32 buffer — host writes f32 data via bitcast, enabling CAS-based f32 add.
@group(0) @binding(3) var<storage, read_write> output: array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.outer_size * params.scatter_size * params.inner_size) {
        return;
    }

    // Decompose scatter index
    let inner = idx % params.inner_size;
    let mid = (idx / params.inner_size) % params.scatter_size;
    let outer = idx / (params.scatter_size * params.inner_size);

    // Get the index to add to
    let scatter_idx = indices[mid];

    // Bounds check
    if (scatter_idx >= params.dim_size) {
        return;
    }

    // Calculate output position
    let output_idx = outer * params.dim_size * params.inner_size +
                     scatter_idx * params.inner_size + inner;

    // Bounds check
    if (output_idx >= params.size) {
        return;
    }

    // Atomic f32 addition via CAS loop on i32 bitcast representation.
    // WGSL has no native atomicAdd for f32, so we:
    //   1. Read current bits as i32 via atomicLoad
    //   2. Interpret as f32, add our value, convert back to i32
    //   3. Attempt atomicCompareExchangeWeak — retry if another thread intervened
    // This guarantees correctness for overlapping scatter indices.
    // Note: values are f64 in canonical; downcast converts to f32 for bitcast.
    let val = f32(values[idx]);
    var old_bits = atomicLoad(&output[output_idx]);
    loop {
        let old_f32 = bitcast<f32>(old_bits);
        let new_f32 = old_f32 + val;
        let new_bits = bitcast<i32>(new_f32);
        let result = atomicCompareExchangeWeak(&output[output_idx], old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
        old_bits = result.old_value;
    }
}
