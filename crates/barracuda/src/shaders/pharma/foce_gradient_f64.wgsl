// SPDX-License-Identifier: AGPL-3.0-only
// FOCE (First-Order Conditional Estimation) per-subject gradient — GPU kernel.
//
// Each thread computes the gradient of the objective function for one subject
// in the population PK model. The per-subject gradient is independent, making
// this embarrassingly parallel.
//
// Input: per-subject observation/prediction residuals and their Jacobians.
// Output: per-subject gradient contribution to the population objective.
//
// Provenance: healthSpring V14 → barraCuda absorption (Mar 2026)

struct FoceConfig {
    n_subjects: u32,
    n_obs_max: u32,
    n_params: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> config: FoceConfig;
@group(0) @binding(1) var<storage, read> residuals: array<f64>;
@group(0) @binding(2) var<storage, read> variances: array<f64>;
@group(0) @binding(3) var<storage, read> jacobian: array<f64>;
@group(0) @binding(4) var<storage, read> obs_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> gradients: array<f64>;
@group(0) @binding(6) var<storage, read_write> objectives: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let subj = gid.x;
    if subj >= config.n_subjects { return; }

    let n_obs = obs_counts[subj];
    let r_base = subj * config.n_obs_max;
    let j_base = subj * config.n_obs_max * config.n_params;
    let g_base = subj * config.n_params;

    // Per-subject FOCE objective: Σ_j [r_j²/v_j + ln(v_j)]
    var obj: f64 = 0.0;
    for (var j: u32 = 0u; j < n_obs; j = j + 1u) {
        let r = residuals[r_base + j];
        let v = variances[r_base + j];
        obj = obj + r * r / v + log(v);
    }
    objectives[subj] = obj;

    // Per-subject gradient: ∂obj/∂θ = Σ_j [2·r_j/v_j · ∂r_j/∂θ]
    for (var p: u32 = 0u; p < config.n_params; p = p + 1u) {
        var grad: f64 = 0.0;
        for (var j: u32 = 0u; j < n_obs; j = j + 1u) {
            let r = residuals[r_base + j];
            let v = variances[r_base + j];
            let jac = jacobian[j_base + j * config.n_params + p];
            grad = grad + 2.0 * r / v * jac;
        }
        gradients[g_base + p] = grad;
    }
}
