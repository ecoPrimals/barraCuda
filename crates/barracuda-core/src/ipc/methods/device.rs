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
                "backend": "coralReef+toadStool",
            }));
        }
    }

    JsonRpcResponse::success(id, serde_json::json!({ "devices": devices }))
}

/// `barracuda.device.probe` — Probe device capabilities.
pub(super) async fn probe(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "available": false,
                "reason": "No GPU device initialized"
            }),
        );
    };

    let limits = dev.device().limits();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "available": true,
            "max_buffer_size": limits.max_buffer_size,
            "max_storage_buffers_per_shader_stage": limits.max_storage_buffers_per_shader_stage,
            "max_compute_workgroup_size_x": limits.max_compute_workgroup_size_x,
            "max_compute_workgroups_per_dimension": limits.max_compute_workgroups_per_dimension,
        }),
    )
}
