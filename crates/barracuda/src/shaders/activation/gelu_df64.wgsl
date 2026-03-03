// GELU activation in DF64 precision — approximate via tanh
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

@group(0) @binding(0) var<storage, read> input_hi: array<f32>;
@group(0) @binding(1) var<storage, read> input_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lo: array<f32>;
@group(0) @binding(4) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }

    let x = Df64(input_hi[idx], input_lo[idx]);
    let half = df64_from_f32(0.5);
    let one = df64_from_f32(1.0);
    let coeff = df64_from_f32(0.044715);
    let sqrt_2_over_pi = df64_from_f32(0.7978845608);

    let x2 = df64_mul(x, x);
    let x3 = df64_mul(x2, x);
    let inner = df64_mul(sqrt_2_over_pi, df64_add(x, df64_mul(coeff, x3)));
    let tanh_val = tanh_df64(inner);
    let result = df64_mul(half, df64_mul(x, df64_add(one, tanh_val)));

    output_hi[idx] = result.hi;
    output_lo[idx] = result.lo;
}
