// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nautilus evolutionary reservoir computing — server-session IPC (Path B).
//!
//! Implements the `job_id`-style session pattern from the Stateful API
//! Architecture Advisory. Each `nautilus.create` call returns a `session_id`;
//! subsequent calls (`observe`, `train`, `predict`, `export`) reference that
//! session by ID. Sessions are server-managed with in-process lifetime.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use barracuda::nautilus::brain::{BetaObservation, NautilusBrain, NautilusBrainConfig};
use barracuda::nautilus::shell::ShellConfig;
use serde_json::Value;

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};

/// Global session store for Nautilus brains (Path B: server-managed state).
static SESSIONS: LazyLock<RwLock<HashMap<String, NautilusBrain>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("nautilus-{ts:x}")
}

/// `nautilus.create` — create a new Nautilus brain session.
///
/// Params:
/// - `name` (string, optional): instance name
/// - `pop_size` (integer, optional, default 8)
/// - `generations_per_train` (integer, optional, default 20)
/// - `min_observations` (integer, optional, default 5)
///
/// Returns `session_id` for subsequent calls.
pub(super) fn nautilus_create(params: &Value, id: Value) -> JsonRpcResponse {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("ipc-session");

    #[expect(clippy::cast_possible_truncation, reason = "pop_size is small")]
    let pop_size = params.get("pop_size").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
    #[expect(clippy::cast_possible_truncation, reason = "generations count")]
    let generations_per_train = params
        .get("generations_per_train")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    #[expect(clippy::cast_possible_truncation, reason = "observation count")]
    let min_observations = params
        .get("min_observations")
        .and_then(|v| v.as_u64())
        .unwrap_or(5) as usize;

    let config = NautilusBrainConfig {
        shell_config: ShellConfig {
            pop_size,
            ..ShellConfig::default()
        },
        generations_per_train,
        min_observations,
    };

    let brain = NautilusBrain::new(config, name);
    let session_id = generate_session_id();

    let Ok(mut sessions) = SESSIONS.write() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "Session store lock poisoned");
    };
    sessions.insert(session_id.clone(), brain);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "session_id": session_id,
            "name": name,
            "pop_size": pop_size,
        }),
    )
}

/// `nautilus.observe` — record an observation into a session.
///
/// Params:
/// - `session_id` (string): session handle
/// - `beta` (number): gauge coupling
/// - `plaquette` (number): average plaquette
/// - `cg_iters` (number): CG iterations
/// - `acceptance` (number): Metropolis acceptance
/// - `delta_h_abs` (number): |ΔH|
/// - `quenched_plaq` (number, optional)
/// - `anderson_r` (number, optional)
pub(super) fn nautilus_observe(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(session_id) = params.get("session_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: session_id");
    };

    let obs = BetaObservation {
        beta: params.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0),
        plaquette: params
            .get("plaquette")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        cg_iters: params
            .get("cg_iters")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        acceptance: params
            .get("acceptance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        delta_h_abs: params
            .get("delta_h_abs")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        quenched_plaq: params.get("quenched_plaq").and_then(|v| v.as_f64()),
        quenched_plaq_var: params.get("quenched_plaq_var").and_then(|v| v.as_f64()),
        anderson_r: params.get("anderson_r").and_then(|v| v.as_f64()),
        anderson_lambda_min: params.get("anderson_lambda_min").and_then(|v| v.as_f64()),
    };

    let Ok(mut sessions) = SESSIONS.write() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "Session store lock poisoned");
    };
    let Some(brain) = sessions.get_mut(session_id) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Unknown session_id: {session_id}"),
        );
    };

    brain.observe(obs);
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "session_id": session_id,
            "observations": brain.observations.len(),
        }),
    )
}

/// `nautilus.train` — evolve the shell and train on accumulated observations.
///
/// Params: `session_id` (string).
/// Returns MSE if training was performed (needs `min_observations`).
pub(super) fn nautilus_train(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(session_id) = params.get("session_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: session_id");
    };

    let Ok(mut sessions) = SESSIONS.write() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "Session store lock poisoned");
    };
    let Some(brain) = sessions.get_mut(session_id) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Unknown session_id: {session_id}"),
        );
    };

    match brain.train() {
        Some(mse) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "session_id": session_id,
                "trained": true,
                "mse": mse,
                "observations": brain.observations.len(),
                "is_drifting": brain.is_drifting(),
            }),
        ),
        None => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "session_id": session_id,
                "trained": false,
                "reason": format!("need at least {} observations", brain.config.min_observations),
                "observations": brain.observations.len(),
            }),
        ),
    }
}

/// `nautilus.predict` — predict observables for a given β.
///
/// Params: `session_id` (string), `beta` (number), `quenched_plaq` (number, optional).
pub(super) fn nautilus_predict(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(session_id) = params.get("session_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: session_id");
    };
    let Some(beta) = params.get("beta").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: beta (number)");
    };
    let quenched_plaq = params.get("quenched_plaq").and_then(|v| v.as_f64());

    let Ok(sessions) = SESSIONS.read() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "Session store lock poisoned");
    };
    let Some(brain) = sessions.get(session_id) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Unknown session_id: {session_id}"),
        );
    };

    match brain.predict_dynamical(beta, quenched_plaq) {
        Some((cg_iters, plaquette, acceptance)) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "session_id": session_id,
                "beta": beta,
                "cg_iters": cg_iters,
                "plaquette": plaquette,
                "acceptance": acceptance,
            }),
        ),
        None => JsonRpcResponse::error(
            id,
            INTERNAL_ERROR,
            "Prediction unavailable (brain not trained or insufficient population)",
        ),
    }
}

/// `nautilus.export` — export session state as JSON for persistence.
///
/// Params: `session_id` (string).
/// Returns the full brain JSON for client-side storage.
pub(super) fn nautilus_export(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(session_id) = params.get("session_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: session_id");
    };

    let Ok(sessions) = SESSIONS.read() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "Session store lock poisoned");
    };
    let Some(brain) = sessions.get(session_id) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Unknown session_id: {session_id}"),
        );
    };

    match brain.to_json() {
        Ok(json) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "session_id": session_id,
                "brain_json": json,
                "observations": brain.observations.len(),
                "trained": brain.trained,
            }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Export failed: {e}")),
    }
}

/// `nautilus.import` — restore a brain session from previously exported JSON.
///
/// Params: `brain_json` (string): JSON from `nautilus.export`.
/// Returns a new `session_id` for the restored brain.
pub(super) fn nautilus_import(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(brain_json) = params.get("brain_json").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: brain_json (string)",
        );
    };

    let brain = match NautilusBrain::from_json(brain_json) {
        Ok(b) => b,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Failed to parse brain_json: {e}"),
            );
        }
    };

    let session_id = generate_session_id();
    let observations = brain.observations.len();
    let trained = brain.trained;

    let Ok(mut sessions) = SESSIONS.write() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "Session store lock poisoned");
    };
    sessions.insert(session_id.clone(), brain);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "session_id": session_id,
            "observations": observations,
            "trained": trained,
        }),
    )
}
