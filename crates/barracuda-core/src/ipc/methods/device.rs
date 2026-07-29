// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device enumeration and probe handlers.

use super::super::jsonrpc::JsonRpcResponse;
use crate::BarraCudaPrimal;
use serde_json::Value;

/// `barracuda.device.list` — Enumerate available compute devices.
pub(super) async fn list(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let mut devices = Vec::new();

    if let Some(cd) = primal.compute_device() {
        if let Some(dev) = cd.wgpu_device() {
            let info = dev.adapter_info();
            devices.push(serde_json::json!({
                "name": info.name,
                "vendor": info.vendor,
                "device_type": format!("{:?}", info.device_type),
                "backend": format!("{:?}", info.backend),
                "driver": info.driver,
                "driver_info": info.driver_info,
            }));
        } else {
            devices.push(serde_json::json!({
                "name": cd.name(),
                "device_type": "SovereignIPC",
                "backend": "sovereign-compile+sovereign-dispatch",
            }));
        }
    }

    JsonRpcResponse::success(id, serde_json::json!({ "devices": devices }))
}

/// `barracuda.device.pool` — Multi-GPU pool status and per-device diagnostics.
pub(super) async fn pool(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    if let Some(gpu_pool) = primal.gpu_pool() {
        let device_status: Vec<Value> = gpu_pool
            .devices()
            .iter()
            .map(|info| {
                serde_json::json!({
                    "name": info.name.as_ref(),
                    "device_class": format!("{:?}", info.device_class),
                    "estimated_gflops": info.estimated_gflops,
                    "vram_gb": info.vram_bytes / (1024 * 1024 * 1024),
                    "is_discrete": info.is_discrete,
                    "f64_builtins": info.f64_builtins_available,
                    "allocations": info.allocation_count(),
                    "allocated_bytes": info.allocated_bytes(),
                    "usage_percent": info.usage_percent(),
                })
            })
            .collect();

        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "available": true,
                "device_count": gpu_pool.device_count(),
                "summary": gpu_pool.summary(),
                "devices": device_status,
            }),
        );
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "available": false,
            "reason": "Multi-GPU pool not initialized (no GPUs or probe skipped)"
        }),
    )
}

/// `barracuda.device.probe` — Probe device capabilities.
pub(super) async fn probe(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    if let Some(dev) = primal.device() {
        let limits = dev.device().limits();
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "available": true,
                "backend": "wgpu",
                "max_buffer_size": limits.max_buffer_size,
                "max_storage_buffers_per_shader_stage": limits.max_storage_buffers_per_shader_stage,
                "max_compute_workgroup_size_x": limits.max_compute_workgroup_size_x,
                "max_compute_workgroups_per_dimension": limits.max_compute_workgroups_per_dimension,
            }),
        );
    }

    if primal.has_sovereign_dispatch() {
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "available": true,
                "backend": "sovereign_ipc",
                "note": "GPU compute via IPC dispatch to peer — limits determined by peer hardware",
            }),
        );
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "available": false,
            "reason": "No GPU device initialized"
        }),
    )
}
