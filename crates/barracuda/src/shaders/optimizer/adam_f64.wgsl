// Adam Optimizer: Adaptive moment estimation optimizer (f64 canonical)
// CUDA equivalent: Custom kernels or cuDNN optimizers
// Formula: Adaptive learning rate with momentum and RMSprop
// Use cases: Deep learning training, state-of-the-art optimization

@group(0) @binding(0) var<storage, read> gradients: array<f64>;      // Input gradients
@group(0) @binding(1) var<storage, read_write> params: array<f64>;   // Model parameters to update
@group(0) @binding(2) var<storage, read_write> m: array<f64>;        // First moment (momentum)
@group(0) @binding(3) var<storage, read_write> v: array<f64>;        // Second moment (RMSprop)

struct AdamParams {
    num_params: u32,
    learning_rate: f64,
    beta1: f64,          // Exponential decay rate for first moment (typically 0.9)
    beta2: f64,          // Exponential decay rate for second moment (typically 0.999)
    epsilon: f64,        // Small constant for numerical stability (typically 1e-8)
    weight_decay: f64,   // L2 regularization strength (typically 0 or 0.01)
    step: u32,           // Current training step (for bias correction)
}
@group(0) @binding(4) var<uniform> adam_params: AdamParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= adam_params.num_params) {
        return;
    }

    // Get current values
    var grad = gradients[gid];
    let param = params[gid];
    var m_val = m[gid];
    var v_val = v[gid];

    // Apply weight decay (L2 regularization) if enabled
    if (adam_params.weight_decay > 0.0) {
        grad = grad + adam_params.weight_decay * param;
    }

    // Update biased first moment estimate (momentum)
    m_val = adam_params.beta1 * m_val + (1.0 - adam_params.beta1) * grad;

    // Update biased second moment estimate (RMSprop)
    v_val = adam_params.beta2 * v_val + (1.0 - adam_params.beta2) * grad * grad;

    // Compute bias-corrected first moment estimate
    let m_hat = m_val / (1.0 - pow_f64(adam_params.beta1, f64(adam_params.step)));

    // Compute bias-corrected second moment estimate
    let v_hat = v_val / (1.0 - pow_f64(adam_params.beta2, f64(adam_params.step)));

    // Update parameters
    let update = adam_params.learning_rate * m_hat / (sqrt_f64(v_hat) + adam_params.epsilon);
    params[gid] = param - update;

    // Store updated moments
    m[gid] = m_val;
    v[gid] = v_val;
}
