// SPDX-License-Identifier: AGPL-3.0-or-later
// RAdam - Rectified Adam Optimizer (f64 canonical)
// Addresses variance warmup issue in Adam
// Automatically adjusts learning rate based on variance tractability
//
// Algorithm:
// 1. Update momentum: m_t = β1 * m_{t-1} + (1 - β1) * g_t
// 2. Update variance: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
// 3. Compute SMA length: ρ_t = ρ_∞ - 2t * β2^t / (1 - β2^t)
// 4. If ρ_t > 5: adaptive update, else: momentum-only update

struct Params {
    size: u32,
    step: u32,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> parameters: array<f64>;
@group(0) @binding(2) var<storage, read> gradients: array<f64>;
@group(0) @binding(3) var<storage, read_write> momentum: array<f64>;
@group(0) @binding(4) var<storage, read_write> variance: array<f64>;
@group(0) @binding(5) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let grad = gradients[idx];
    let param = parameters[idx];

    // Update biased first moment estimate
    let m = params.beta1 * momentum[idx] + (1.0 - params.beta1) * grad;
    momentum[idx] = m;

    // Update biased second raw moment estimate
    let v = params.beta2 * variance[idx] + (1.0 - params.beta2) * grad * grad;
    variance[idx] = v;

    // Compute bias correction for first moment
    let step_f = f64(params.step);
    let beta1_pow = pow_f64(params.beta1, step_f);
    let beta2_pow = pow_f64(params.beta2, step_f);
    let m_hat = m / (1.0 - beta1_pow);

    // Compute maximum length of approximated SMA
    let rho_inf = 2.0 / (1.0 - params.beta2) - 1.0;

    // Compute current approximated SMA length
    let rho_t = rho_inf - 2.0 * step_f * beta2_pow / (1.0 - beta2_pow);

    var update: f64;
    if (rho_t > 5.0) {
        // Variance is tractable, use adaptive learning rate
        let v_hat = v / (1.0 - beta2_pow);
        let r_num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
        let r_den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
        let r = sqrt_f64(r_num / r_den);
        update = params.learning_rate * r * m_hat / (sqrt_f64(v_hat) + params.epsilon);
    } else {
        // Variance not tractable, use unadapted step (momentum only)
        update = params.learning_rate * m_hat;
    }

    output[idx] = param - update;
}
