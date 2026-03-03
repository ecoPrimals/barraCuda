// Mask Convert - Convert f32 mask to u32 mask (0 or 1) (f64 canonical)
// Converts f64 mask values to u32 boolean mask (1 if non-zero, 0 if zero)
//
// Used by nonzero, masked_select, and other operations that need boolean masks

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_mask: array<f64>;  // f64 mask input
@group(0) @binding(2) var<storage, read_write> output_mask: array<u32>;  // u32 boolean mask output

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    
    // Convert f64 to u32 boolean: 1 if non-zero, 0 if zero
    output_mask[idx] = select(0u, 1u, input_mask[idx] != 0.0);
}
