// SPDX-License-Identifier: AGPL-3.0-or-later
//! Named compute dispatch, shader binary execution, and shape parsing helpers.
//!
//! Implements barraCuda-native op-based dispatch (`compute.dispatch`) and the
//! cross-gate pipeline methods (`compute.dispatch.capabilities`,
//! `compute.dispatch.submit`, `compute.dispatch.result`). Peer routing is
//! resolved at runtime via capability-based discovery — no compile-time
//! knowledge of which primal serves `compute.dispatch.submit`.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use crate::BarraCudaPrimal;
use serde_json::Value;

/// Parse a JSON array of u64 values into a `Vec<usize>`, returning `None` if
/// any dimension overflows the platform `usize`.
pub(super) fn parse_shape(arr: &[Value]) -> Option<Vec<usize>> {
    arr.iter()
        .filter_map(|v| v.as_u64())
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()
}

/// `barracuda.compute.dispatch` — Dispatch a named compute operation.
///
/// Rather than accepting raw WGSL (which would require shader security auditing),
/// this dispatches named operations from barraCuda's shader library. Pass input
/// data and the operation produces output stored in the tensor store.
///
/// Validates all input parameters before checking device availability so that
/// callers receive precise validation errors regardless of hardware state.
pub(super) async fn compute_dispatch(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(op) = params.get("op").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: op (e.g. 'zeros', 'ones', 'read')",
        );
    };

    match op {
        "zeros" | "ones" => {
            let Some(shape) = params.get("shape").and_then(|v| v.as_array()) else {
                return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: shape");
            };
            let Some(shape_vec) = parse_shape(shape) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    "Shape dimension exceeds platform usize",
                );
            };
            let Some(dev) = primal.device() else {
                return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
            };
            let dev_arc = dev;
            let result = if op == "zeros" {
                barracuda::tensor::Tensor::zeros_on(shape_vec.clone(), dev_arc).await
            } else {
                barracuda::tensor::Tensor::ones_on(shape_vec.clone(), dev_arc).await
            };
            match result {
                Ok(t) => {
                    let tensor_id = primal.store_tensor(t);
                    JsonRpcResponse::success(
                        id,
                        serde_json::json!({
                            "status": "completed", "op": op, "tensor_id": tensor_id, "shape": shape_vec,
                        }),
                    )
                }
                Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("{op} failed: {e}")),
            }
        }
        "read" => {
            let Some(tensor_id) = params.get("tensor_id").and_then(|v| v.as_str()) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    "Missing required param: tensor_id",
                );
            };
            let Some(tensor) = primal.get_tensor(tensor_id) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Tensor not found: {tensor_id}"),
                );
            };
            match tensor.to_vec() {
                Ok(data) => JsonRpcResponse::success(
                    id,
                    serde_json::json!({
                        "status": "completed", "tensor_id": tensor_id,
                        "shape": tensor.shape(), "data": data,
                    }),
                ),
                Err(e) => {
                    JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Readback failed: {e}"))
                }
            }
        }
        _ => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Unknown op: {op}. Available: zeros, ones, read"),
        ),
    }
}

// ─── Cross-gate pipeline methods ─────────────────────────────────────────────
// These implement the cross-gate dispatch contract. Any peer primal advertising
// `compute.dispatch.submit` is resolved at runtime via capability-based discovery.

/// Outcome of a dispatch job — typed for compile-time correctness.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DispatchStatus {
    Completed,
    Failed,
}

/// Stored result of a dispatch job.
#[derive(Clone)]
pub(crate) struct DispatchJob {
    pub(crate) status: DispatchStatus,
    pub(crate) output: Option<Value>,
    pub(crate) error: Option<String>,
}

fn generate_job_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let hash = blake3::hash(format!("dispatch-{ts}-{seq}").as_bytes());
    format!("job-{}", &hash.to_hex()[..16])
}

/// `compute.dispatch.capabilities` — Report GPU compute capabilities.
///
/// Returns the set of capabilities this primal can serve for cross-gate
/// dispatch routing. Peer primals resolve this at runtime via
/// `ipc.resolve` to decide where to route shader workloads.
pub(super) fn dispatch_capabilities(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let mut capabilities: Vec<&str> = Vec::new();

    let has_gpu = primal.device().is_some();
    let has_f64 = primal.compute_device().is_some_and(|d| d.has_f64_shaders());
    let has_spirv = primal.device().is_some_and(|d| d.has_spirv_passthrough());

    capabilities.push("gpu.f32");
    if has_gpu {
        capabilities.push("gpu.tensor_ops");
        capabilities.push("gpu.dispatch_submit");
    }
    if has_f64 {
        capabilities.push("gpu.f64");
        capabilities.push("gpu.df64");
    }
    if has_spirv {
        capabilities.push("gpu.spirv_passthrough");
    }
    capabilities.push("cpu.tensor_ops");
    capabilities.push("cpu.math");
    capabilities.push("cpu.stats");

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "capabilities": capabilities,
            "gpu_available": has_gpu,
            "f64_shaders": has_f64,
            "primal": "barraCuda",
            "version": env!("CARGO_PKG_VERSION"),
        }),
    )
}

/// `compute.dispatch.submit` — Accept a shader or tensor workload.
///
/// Two modes of operation:
///
/// ## Mode 1: Shader binary dispatch (when `binary_b64` is present)
///
/// Callers send compiled SPIR-V/PTX binaries with:
/// - `binary_b64`: base64-encoded compiled shader binary
/// - `input`: `{"data": [f64], "format": "f64_array"}`
/// - `input_hash`: BLAKE3 hex of serialized input
/// - `shader_info`: compiler metadata (workgroup size, bindings, etc.)
/// - `bindings`: buffer binding descriptors
/// - `dispatch_mode`: "passthrough"
/// - `bdf`: optional PCI BDF for GPU targeting
/// - `spring`: calling spring name
///
/// When a binary is present, barraCuda resolves the `compute.dispatch.submit`
/// capability via `ipc.resolve` and forwards the full request to the
/// resolved peer. If no peer is available, falls back to tensor passthrough
/// with `"routed": false` in the response.
///
/// ## Mode 2: Tensor passthrough (no `binary_b64`)
///
/// When no shader binary is provided, treats `input.data` as raw tensor
/// data: uploads to GPU, reads back, stores in job store.
///
/// ## Returns
///
/// `{"job_id": str, "status": "completed"|"failed", "routed": bool, ...}`
pub(super) async fn dispatch_submit(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let has_binary = params.get("binary_b64").is_some();

    if has_binary {
        return dispatch_submit_shader(primal, params, id).await;
    }

    dispatch_submit_tensor(primal, params, id).await
}

/// Shader binary dispatch — attempt capability-based routing, fall back to tensor passthrough.
async fn dispatch_submit_shader(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    if let Some(forwarded) = try_forward_to_dispatch_peer(params).await {
        let job_id = generate_job_id();
        let job = DispatchJob {
            status: DispatchStatus::Completed,
            output: Some(forwarded.result),
            error: None,
        };
        primal.store_dispatch_job(job_id.clone(), job);
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "job_id": job_id,
                "status": "completed",
                "routed": true,
                "routed_to": forwarded.peer_socket,
            }),
        );
    }

    tracing::debug!(
        "compute.dispatch.submit: binary_b64 present but no peer advertising \
         compute.dispatch.submit — falling back to tensor passthrough"
    );

    let input_data = params
        .get("input")
        .and_then(|i| i.get("data"))
        .and_then(|d| d.as_array());

    let Some(data_arr) = input_data else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Shader binary provided but no dispatch peer available, and no input.data \
             for tensor fallback. Deploy a peer with compute.dispatch.submit or provide input.data.",
        );
    };

    let mut resp = dispatch_submit_tensor_inner(primal, data_arr).await;
    if let Some(ref mut output) = resp.output {
        if let Some(obj) = output.as_object_mut() {
            obj.insert("routed".into(), Value::Bool(false));
            obj.insert(
                "note".into(),
                Value::String(
                    "binary_b64 ignored — no dispatch peer available; \
                     tensor passthrough used"
                        .into(),
                ),
            );
        }
    }

    let job_id = generate_job_id();
    let is_completed = resp.status == DispatchStatus::Completed;
    let error = resp.error.clone();
    let n = data_arr.len();

    primal.store_dispatch_job(job_id.clone(), resp);

    if is_completed {
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "job_id": job_id,
                "status": "completed",
                "routed": false,
                "elements": n,
                "note": "binary_b64 present but no dispatch peer available — tensor passthrough used",
            }),
        )
    } else {
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "job_id": job_id,
                "status": "failed",
                "routed": false,
                "error": error,
            }),
        )
    }
}

/// Tensor passthrough dispatch — upload input to GPU, read back, store result.
async fn dispatch_submit_tensor(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let input_data = params
        .get("input")
        .and_then(|i| i.get("data"))
        .and_then(|d| d.as_array());

    let Some(data_arr) = input_data else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: input.data (array of numbers)",
        );
    };

    let result = dispatch_submit_tensor_inner(primal, data_arr).await;
    let job_id = generate_job_id();
    let is_completed = result.status == DispatchStatus::Completed;
    let error = result.error.clone();
    let n = data_arr.len();

    primal.store_dispatch_job(job_id.clone(), result);

    if is_completed {
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "job_id": job_id,
                "status": "completed",
                "routed": false,
                "elements": n,
            }),
        )
    } else {
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "job_id": job_id,
                "status": "failed",
                "error": error,
            }),
        )
    }
}

/// Core tensor passthrough logic shared by both shader-fallback and tensor modes.
async fn dispatch_submit_tensor_inner(primal: &BarraCudaPrimal, data_arr: &[Value]) -> DispatchJob {
    let data: Vec<f32> = data_arr
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    if data.is_empty() {
        return DispatchJob {
            status: DispatchStatus::Failed,
            output: None,
            error: Some("input.data must be non-empty".into()),
        };
    }

    let n = data.len();

    if let Some(dev) = primal.device() {
        match barracuda::tensor::Tensor::from_data(&data, vec![n], dev) {
            Ok(tensor) => match tensor.to_vec() {
                Ok(output) => DispatchJob {
                    status: DispatchStatus::Completed,
                    output: Some(serde_json::json!({
                        "output": output,
                        "format": "f32_array",
                        "substrate": "gpu",
                        "shape": [n],
                    })),
                    error: None,
                },
                Err(e) => DispatchJob {
                    status: DispatchStatus::Failed,
                    output: None,
                    error: Some(format!("GPU readback failed: {e}")),
                },
            },
            Err(e) => DispatchJob {
                status: DispatchStatus::Failed,
                output: None,
                error: Some(format!("GPU tensor creation failed: {e}")),
            },
        }
    } else {
        DispatchJob {
            status: DispatchStatus::Completed,
            output: Some(serde_json::json!({
                "output": data,
                "format": "f32_array",
                "substrate": "cpu_fallback",
                "shape": [n],
            })),
            error: None,
        }
    }
}

/// Result of a successful capability-based forward to a dispatch peer.
struct ForwardedResult {
    result: Value,
    peer_socket: String,
}

/// Capability-based forward to any peer advertising `compute.dispatch.submit`.
///
/// Resolution chain:
/// 1. `DISCOVERY_SOCKET` → `ipc.resolve { capability: "compute.dispatch.submit" }`
/// 2. If resolved: forward full params as `compute.dispatch.submit` JSON-RPC
/// 3. Returns the result and the socket path of the peer that answered
///
/// No primal name hardcoding — any peer with the capability is accepted.
async fn try_forward_to_dispatch_peer(params: &Value) -> Option<ForwardedResult> {
    let discovery_path = std::env::var(crate::env_keys::DISCOVERY_SOCKET).ok()?;
    if !std::path::Path::new(&discovery_path).exists() {
        return None;
    }

    let disc_endpoint = sourdough_core::TransportEndpoint::uds(&discovery_path);
    let stream = sourdough_core::connect_transport(&disc_endpoint).await.ok()?;
    let mut buf_reader = tokio::io::BufReader::new(stream);

    let resolve_req = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "ipc.resolve",
        "params": { "capability": "compute.dispatch.submit" },
        "id": 1
    });
    let mut line = serde_json::to_string(&resolve_req).ok()?;
    line.push('\n');
    tokio::io::AsyncWriteExt::write_all(buf_reader.get_mut(), line.as_bytes())
        .await
        .ok()?;

    use tokio::io::AsyncBufReadExt;
    let mut resp_line = String::new();
    buf_reader.read_line(&mut resp_line).await.ok()?;
    let resp: Value = serde_json::from_str(resp_line.trim()).ok()?;

    let result = resp.get("result")?;
    let unix_addr = result
        .get("unix")
        .or_else(|| result.get("socket"))
        .and_then(|v| v.as_str())?;
    let sock_path = std::path::PathBuf::from(unix_addr.trim_start_matches("unix://"));
    if !sock_path.exists() {
        return None;
    }

    let peer_socket = sock_path.to_string_lossy().into_owned();

    let peer_endpoint = sourdough_core::TransportEndpoint::uds(&peer_socket);
    let peer_stream = sourdough_core::connect_transport(&peer_endpoint).await.ok()?;
    let mut peer_buf = tokio::io::BufReader::new(peer_stream);

    let forward_req = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "compute.dispatch.submit",
        "params": params,
        "id": 1
    });
    let mut fwd_line = serde_json::to_string(&forward_req).ok()?;
    fwd_line.push('\n');
    tokio::io::AsyncWriteExt::write_all(peer_buf.get_mut(), fwd_line.as_bytes())
        .await
        .ok()?;

    let mut peer_resp_line = String::new();
    peer_buf.read_line(&mut peer_resp_line).await.ok()?;
    let peer_resp: Value = serde_json::from_str(peer_resp_line.trim()).ok()?;

    let forwarded_result = peer_resp.get("result").cloned()?;
    Some(ForwardedResult {
        result: forwarded_result,
        peer_socket,
    })
}

/// `compute.dispatch.result` — Retrieve output data for a previously submitted job.
///
/// Wire contract: `{"job_id": str}` → output JSON or error.
pub(super) fn dispatch_result(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(job_id) = params.get("job_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: job_id");
    };

    match primal.get_dispatch_job(job_id) {
        Some(job) if job.status == DispatchStatus::Completed => {
            let output = job.output.unwrap_or(Value::Null);
            JsonRpcResponse::success(id, output)
        }
        Some(job) => JsonRpcResponse::error(
            id,
            INTERNAL_ERROR,
            format!(
                "Job {job_id} failed: {}",
                job.error.as_deref().unwrap_or("unknown error")
            ),
        ),
        None => JsonRpcResponse::error(id, INVALID_PARAMS, format!("Job not found: {job_id}")),
    }
}
