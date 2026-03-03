// local_response_norm.wgsl - Local Response Normalization (LRN) (f64 canonical)
//
// Normalizes activations within local neighborhoods
// Used in AlexNet and other early CNNs
//
// Formula: y_i = x_i / (k + alpha * sum(x_j^2) / size)^beta

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    size: u32,      // Neighborhood size
    alpha: f64,     // Scaling parameter (typically 1e-4)
    beta: f64,      // Exponent (typically 0.75)
    k: f64,         // Bias (typically 1.0 or 2.0)
}

@group(0) @binding(0) var<storage, read> input: array<f64>;         // [B, C, H, W]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // [B, C, H, W]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let h = global_id.y;
    let w = global_id.x;

    if (c >= params.channels || h >= params.height || w >= params.width) {
        return;
    }

    let half_size = params.size / 2u;

    // Calculate sum of squares in local neighborhood (across channels)
    var sum_sq: f64 = 0.0;

    let c_start = max(0, i32(c) - i32(half_size));
    let c_end = min(i32(params.channels), i32(c) + i32(half_size) + 1);

    for (var nc: i32 = c_start; nc < c_end; nc = nc + 1) {
        let idx = b * params.channels * params.height * params.width +
                  u32(nc) * params.height * params.width +
                  h * params.width +
                  w;

        let val = input[idx];
        sum_sq = sum_sq + val * val;
    }

    // Compute normalization
    let scale = pow_f64(params.k + params.alpha * sum_sq / f64(params.size), params.beta);

    let in_idx = b * params.channels * params.height * params.width +
                 c * params.height * params.width +
                 h * params.width +
                 w;

    output[in_idx] = input[in_idx] / scale;
}
