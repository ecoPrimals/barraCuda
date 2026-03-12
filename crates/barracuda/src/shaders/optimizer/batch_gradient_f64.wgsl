// SPDX-License-Identifier: AGPL-3.0-only
// Batch Numerical Gradient Computation - f64 canonical
//
// Computes ∇f(x) using central differences, FULLY PARALLEL
// ∂f/∂xᵢ ≈ (f(x + εeᵢ) - f(x - εeᵢ)) / (2ε)

struct Params {
    n: u32,
    epsilon: f64,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f64>;
@group(0) @binding(2) var<storage, read> f_plus: array<f64>;
@group(0) @binding(3) var<storage, read> f_minus: array<f64>;
@group(0) @binding(4) var<storage, read_write> gradient: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn central_difference(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i >= params.n) {
        return;
    }

    gradient[i] = (f_plus[i] - f_minus[i]) / (2.0 * params.epsilon);
}

@compute @workgroup_size(256, 1, 1)
fn forward_difference(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i >= params.n) {
        return;
    }

    let f_x = f_minus[0];
    gradient[i] = (f_plus[i] - f_x) / params.epsilon;
}

struct PerturbParams {
    n: u32,
    epsilon: f64,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> perturb_params: PerturbParams;
@group(0) @binding(1) var<storage, read> base_x: array<f64>;
@group(0) @binding(2) var<storage, read_write> x_perturbed: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn generate_perturbed_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = perturb_params.n;

    let point_idx = idx / n;
    let component = idx % n;

    if (point_idx >= 2u * n) {
        return;
    }

    var val = base_x[component];

    let perturb_dim = point_idx % n;
    let is_plus = point_idx < n;

    if (component == perturb_dim) {
        if (is_plus) {
            val = val + perturb_params.epsilon;
        } else {
            val = val - perturb_params.epsilon;
        }
    }

    x_perturbed[idx] = val;
}
