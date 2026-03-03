// Complex Sqrt: polar form

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
struct Params { num_complex: u32, }

fn complex_sqrt(z: vec2<f32>) -> vec2<f32> {
    let r = length(z);
    let theta = atan2(z.y, z.x);
    let sqrt_r = sqrt(r);
    let half_theta = theta * 0.5;
    return vec2<f32>(sqrt_r * cos(half_theta), sqrt_r * sin(half_theta));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_complex) { return; }
    let base = idx * 2u;
    let z = vec2<f32>(input[base], input[base + 1u]);
    let result = complex_sqrt(z);
    output[base] = result.x;
    output[base + 1u] = result.y;
}
