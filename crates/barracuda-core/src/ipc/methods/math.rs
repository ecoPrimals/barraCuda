// SPDX-License-Identifier: AGPL-3.0-or-later
//! Math, activation, noise, RNG, and ODE handlers for JSON-RPC IPC.
//!
//! Wires barraCuda's CPU-side math primitives to the IPC surface per
//! `SEMANTIC_METHOD_NAMING_STANDARD.md`. These are lightweight CPU ops
//! suitable for composition graph nodes — GPU tensor ops live in `tensor.rs`.
//!
//! Statistics handlers live in `stats.rs`; signal processing in `signal.rs`.

use super::super::jsonrpc::{INVALID_PARAMS, JsonRpcResponse};
use super::params::{extract_f64, extract_f64_array, extract_matrix};
use serde_json::Value;

// ── Math element-wise ─────────────────────────────────────────────────────

/// `math.sigmoid` — element-wise sigmoid on f64 array.
pub(super) fn math_sigmoid(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result: Vec<f64> = data
        .iter()
        .map(|&x| barracuda::activations::sigmoid(x))
        .collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `math.log2` — element-wise log base 2 on f64 array.
pub(super) fn math_log2(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result: Vec<f64> = data.iter().map(|x| x.log2()).collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

// ── Noise & RNG ───────────────────────────────────────────────────────────

/// `noise.perlin2d` — CPU Perlin noise at (x, y).
pub(super) fn noise_perlin2d(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x");
    };
    let Some(y) = extract_f64(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y");
    };
    let result = barracuda::ops::procedural::perlin_noise::perlin_2d_cpu(x, y);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `noise.perlin3d` — CPU 3D Perlin noise at (x, y, z).
pub(super) fn noise_perlin3d(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x");
    };
    let Some(y) = extract_f64(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y");
    };
    let Some(z) = extract_f64(params, "z") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: z");
    };
    let result = barracuda::ops::procedural::perlin_noise::perlin_3d_cpu(x, y, z);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `rng.uniform` — generate n uniform random f64 values in [min, max).
pub(super) fn rng_uniform(params: &Value, id: Value) -> JsonRpcResponse {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n is a user-provided count; truncation on 32-bit is acceptable (capped at u32::MAX items)"
    )]
    let n = params.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let min = extract_f64(params, "min").unwrap_or(0.0);
    let max = extract_f64(params, "max").unwrap_or(1.0);
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);

    if max <= min {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "max must be > min");
    }

    let result: Vec<f64> = barracuda::rng::uniform_f64_sequence(seed, n)
        .into_iter()
        .map(|v| v.mul_add(max - min, min))
        .collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

// ── Activation functions ──────────────────────────────────────────────────

/// `activation.fitts` — Fitts' law movement time prediction.
pub(super) fn activation_fitts(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(distance) = extract_f64(params, "distance") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: distance");
    };
    let Some(width) = extract_f64(params, "width") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: width");
    };
    if width <= 0.0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "width must be > 0");
    }
    let a = extract_f64(params, "a").unwrap_or(0.0);
    let b = extract_f64(params, "b").unwrap_or(0.155);
    let variant = params
        .get("variant")
        .and_then(|v| v.as_str())
        .unwrap_or("shannon");
    let id_bits = match variant {
        "shannon" => (distance / width + 1.0).log2().max(0.0),
        "fitts" => (2.0 * distance / width).log2().max(0.0),
        other => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Unknown variant: {other}. Expected \"shannon\" or \"fitts\""),
            );
        }
    };
    let mt = b.mul_add(id_bits, a);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": mt, "movement_time": mt, "index_of_difficulty": id_bits, "variant": variant }),
    )
}

/// `activation.hick` — Hick-Hyman law reaction time prediction.
pub(super) fn activation_hick(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(n_choices) = params.get("n_choices").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: n_choices");
    };
    if n_choices == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "n_choices must be > 0");
    }
    let a = extract_f64(params, "a").unwrap_or(0.0);
    let b = extract_f64(params, "b").unwrap_or(0.155);
    let include_no_choice = params
        .get("include_no_choice")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    #[expect(
        clippy::cast_precision_loss,
        reason = "n_choices is user-provided, fits f64 mantissa"
    )]
    let hick_bits = if include_no_choice {
        ((n_choices + 1) as f64).log2()
    } else {
        (n_choices as f64).log2()
    };
    let rt = b.mul_add(hick_bits, a);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": rt, "reaction_time": rt, "information_bits": hick_bits, "include_no_choice": include_no_choice }),
    )
}

/// `activation.softmax` — element-wise softmax on f64 array (exp-normalize, CPU).
pub(super) fn activation_softmax(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    if data.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "data must be non-empty");
    }
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let result: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `activation.gelu` — element-wise GELU on f64 array.
pub(super) fn activation_gelu(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result = barracuda::activations::gelu_batch(&data);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

// ── ODE integration ───────────────────────────────────────────────────────

/// `ode.step` — stateless RK4 integration of a linear ODE system (Path A).
///
/// Accepts the current state, linear system matrix A and forcing vector b
/// (dy/dt = A*y + b), time step, and number of steps. Returns the final state.
pub(super) fn ode_step(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(state) = extract_f64_array(params, "state") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: state (array)");
    };
    let Some(a_matrix) = extract_matrix(params, "a") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: a (n_vars × n_vars matrix)",
        );
    };

    let n_vars = state.len();
    if n_vars == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "state must be non-empty");
    }
    if a_matrix.len() != n_vars {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!(
                "a: row count ({}) must equal state length ({n_vars})",
                a_matrix.len()
            ),
        );
    }
    for (i, row) in a_matrix.iter().enumerate() {
        if row.len() != n_vars {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!(
                    "a[{i}]: column count ({}) must equal state length ({n_vars})",
                    row.len()
                ),
            );
        }
    }

    let b = match extract_f64_array(params, "b") {
        Some(bv) if bv.len() == n_vars => bv,
        Some(bv) => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!(
                    "b: length ({}) must equal state length ({n_vars})",
                    bv.len()
                ),
            );
        }
        None => vec![0.0; n_vars],
    };

    let dt = match params.get("dt").and_then(|v| v.as_f64()) {
        Some(d) if d > 0.0 => d,
        Some(_) => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "dt must be > 0");
        }
        None => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                "Missing required param: dt (positive number)",
            );
        }
    };
    #[expect(clippy::cast_possible_truncation, reason = "n_steps is a step count")]
    let n_steps = params.get("n_steps").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let t0 = params.get("t0").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let deriv = |_t: f64, y: &[f64]| -> Vec<f64> {
        let mut dy = b.clone();
        for (i, row) in a_matrix.iter().enumerate() {
            for (j, &a_ij) in row.iter().enumerate() {
                dy[i] = a_ij.mul_add(y[j], dy[i]);
            }
        }
        dy
    };

    let mut y = state;
    let mut t = t0;

    for _ in 0..n_steps {
        let k1 = deriv(t, &y);
        let y2: Vec<f64> = y
            .iter()
            .zip(&k1)
            .map(|(&yi, &k)| (0.5 * dt).mul_add(k, yi))
            .collect();
        let t_mid = 0.5f64.mul_add(dt, t);
        let k2 = deriv(t_mid, &y2);
        let y3: Vec<f64> = y
            .iter()
            .zip(&k2)
            .map(|(&yi, &k)| (0.5 * dt).mul_add(k, yi))
            .collect();
        let k3 = deriv(t_mid, &y3);
        let y4: Vec<f64> = y
            .iter()
            .zip(&k3)
            .map(|(&yi, &k)| dt.mul_add(k, yi))
            .collect();
        let k4 = deriv(t + dt, &y4);

        let sixth = 1.0 / 6.0;
        for i in 0..n_vars {
            y[i] = (dt * sixth).mul_add(
                2.0f64.mul_add(k3[i], 2.0f64.mul_add(k2[i], k1[i])) + k4[i],
                y[i],
            );
        }
        t += dt;
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({ "state": y, "t_final": t, "n_steps": n_steps }),
    )
}
