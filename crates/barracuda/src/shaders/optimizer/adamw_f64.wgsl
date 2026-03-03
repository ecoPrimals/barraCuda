// AdamW Optimizer: Adam with Decoupled Weight Decay (f64 canonical)
// Key difference from Adam: Weight decay applied AFTER gradient update
// This makes weight decay independent of gradient-based learning rate adaptation
//
// Formula:
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad²
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// param = param - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * wd * param
//
// Used in: BERT, GPT, modern transformers (superior to Adam for large models)

@group(0) @binding(0) var<storage, read> gradients: array<f64>;      // Input gradients
@group(0) @binding(1) var<storage, read_write> params: array<f64>;   // Model parameters to update
@group(0) @binding(2) var<storage, read_write> m: array<f64>;        // First moment (momentum)
@group(0) @binding(3) var<storage, read_write> v: array<f64>;        // Second moment (RMSprop)

struct AdamWParams {
    num_params: u32,
    learning_rate: f64,
    beta1: f64,          // Exponential decay rate for first moment (typically 0.9)
    beta2: f64,          // Exponential decay rate for second moment (typically 0.999)
    epsilon: f64,        // Small constant for numerical stability (typically 1e-8)
    weight_decay: f64,   // Decoupled weight decay (typically 0.01)
    step: u32,           // Current training step (for bias correction)
}
@group(0) @binding(4) var<uniform> adamw_params: AdamWParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= adamw_params.num_params) {
        return;
    }

    // Get current values
    let grad = gradients[gid];
    let param = params[gid];
    var m_val = m[gid];
    var v_val = v[gid];

    // Update biased first moment estimate (momentum)
    m_val = adamw_params.beta1 * m_val + (1.0 - adamw_params.beta1) * grad;

    // Update biased second moment estimate (RMSprop)
    v_val = adamw_params.beta2 * v_val + (1.0 - adamw_params.beta2) * grad * grad;

    // Compute bias-corrected first moment estimate
    let m_hat = m_val / (1.0 - pow_f64(adamw_params.beta1, f64(adamw_params.step)));

    // Compute bias-corrected second moment estimate
    let v_hat = v_val / (1.0 - pow_f64(adamw_params.beta2, f64(adamw_params.step)));

    // Adam update (gradient-based)
    let adam_update = adamw_params.learning_rate * m_hat / (sqrt_f64(v_hat) + adamw_params.epsilon);

    // Decoupled weight decay (KEY DIFFERENCE from Adam!)
    let weight_decay_update = adamw_params.learning_rate * adamw_params.weight_decay * param;

    // Update parameters (Adam update + decoupled weight decay)
    params[gid] = param - adam_update - weight_decay_update;

    // Store updated moments
    m[gid] = m_val;
    v[gid] = v_val;
}
