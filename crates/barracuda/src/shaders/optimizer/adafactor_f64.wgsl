// adafactor.wgsl - Adafactor Optimizer (f64 canonical)
//
// Memory-efficient adaptive learning rate optimizer
// Reduces memory by factorizing second moment matrix
//
// Reference: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" by Shazeer & Stern (2018)

struct Params {
    size: u32,
    lr: f64,
    beta1: f64,          // Exponential decay rate for first moment (0 = disable)
    beta2: f64,          // Exponential decay rate for second moment
    epsilon1: f64,       // Regularization for RMS
    epsilon2: f64,       // Regularization for parameter scale
    clip_threshold: f64, // Gradient clipping threshold
    decay_rate: f64,     // -0.8 for typical use
    step: u32,
}

@group(0) @binding(0) var<storage, read> grad: array<f64>;           // Gradients
@group(0) @binding(1) var<storage, read_write> param: array<f64>;    // Parameters
@group(0) @binding(2) var<storage, read_write> m: array<f64>;        // First moment (optional)
@group(0) @binding(3) var<storage, read_write> v: array<f64>;        // Second moment (factored)
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let g = grad[idx];
    let p = param[idx];

    // Compute RMS of gradient
    let g_sq = g * g;

    // Update second moment (simplified - full version uses row/column factorization)
    v[idx] = params.beta2 * v[idx] + (1.0 - params.beta2) * g_sq;

    // Compute RMS
    let rms = sqrt_f64(v[idx] + params.epsilon1);

    // Relative step size
    let step_f = f64(params.step);
    let rho_t = min(params.lr, 1.0 / sqrt_f64(step_f));

    // Parameter scale
    let param_scale = max(params.epsilon2, sqrt_f64(p * p));

    // Adaptive learning rate
    let adaptive_lr = rho_t / rms * param_scale;

    // Gradient clipping
    var clipped_g = g;
    if (params.clip_threshold > 0.0) {
        let grad_norm = abs(g);
        if (grad_norm > params.clip_threshold) {
            clipped_g = clipped_g * params.clip_threshold / grad_norm;
        }
    }

    // Update with or without momentum
    var update: f64;
    if (params.beta1 > 0.0) {
        m[idx] = params.beta1 * m[idx] + (1.0 - params.beta1) * clipped_g;
        update = adaptive_lr * m[idx];
    } else {
        update = adaptive_lr * clipped_g;
    }

    // Update parameters
    param[idx] = p - update;
}
