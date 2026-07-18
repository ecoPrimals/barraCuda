// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC wire protocol for sovereign compute dispatch submission.
//!
//! Extracted from `sovereign_device.rs` as a distinct domain: this module handles
//! the IPC transport (TCP + HTTP/1.1 + JSON-RPC 2.0) for submitting compiled
//! binaries to the `compute.dispatch` primal. The parent module handles device
//! lifecycle, compilation orchestration, and GpuBackend trait implementation.

use crate::error::{BarracudaError, Result};

use super::backend::HardwareHint;

/// Metadata about a compiled shader needed for GPU dispatch.
///
/// Sent to the dispatch primal so it can configure QMD (NVIDIA) or PM4 (AMD)
/// descriptors with the correct register count, workgroup dimensions, shared
/// memory size, and barrier count.
#[derive(Debug, Clone, Copy)]
pub(super) struct ShaderDispatchInfo {
    pub gpr_count: u32,
    pub workgroup: [u32; 3],
    pub shared_mem_bytes: u32,
    pub barrier_count: u32,
}

/// Serialisable buffer binding descriptor for IPC compute dispatch.
#[derive(Debug, Clone)]
pub(super) struct IpcBufferBinding {
    pub index: u32,
    pub buffer_id: u64,
    pub size: u64,
    pub read_only: bool,
}

/// Submit a dispatch request to the `compute.dispatch` primal via JSON-RPC.
///
/// Sends the compiled native binary, workgroup dimensions, buffer binding
/// descriptors (IDs + access mode), and hardware routing hint. The dispatch
/// primal maps buffer IDs to GPU memory and routes to the appropriate
/// hardware unit.
pub(super) fn submit_dispatch(
    dispatch_addr: &str,
    staged_buffers: &std::sync::Mutex<std::collections::HashMap<u64, bytes::BytesMut>>,
    binary: &[u8],
    workgroups: (u32, u32, u32),
    bindings: &[IpcBufferBinding],
    hardware_hint: HardwareHint,
    shader_info: ShaderDispatchInfo,
) -> Result<()> {
    let binding_descriptors: Vec<serde_json::Value> = bindings
        .iter()
        .map(|b| {
            serde_json::json!({
                "index": b.index,
                "buffer_id": b.buffer_id,
                "size": b.size,
                "read_only": b.read_only,
            })
        })
        .collect();

    let hint_str = match hardware_hint {
        HardwareHint::Compute => "compute",
        HardwareHint::TensorCore => "tensor_core",
        HardwareHint::RtCore => "rt_core",
        HardwareHint::ZBuffer => "zbuffer",
        HardwareHint::TextureUnit => "texture_unit",
        HardwareHint::RopBlend => "rop_blend",
    };

    let buffer_data: Vec<serde_json::Value> = bindings
        .iter()
        .filter_map(|b| {
            let staged = staged_buffers.lock().ok()?;
            let data = staged.get(&b.buffer_id)?;
            Some(serde_json::json!({
                "buffer_id": b.buffer_id,
                "data": data.to_vec(),
            }))
        })
        .collect();

    let request = serde_json::json!({
        "binary": binary,
        "workgroup_size": [workgroups.0, workgroups.1, workgroups.2],
        "bindings": binding_descriptors,
        "hardware_hint": hint_str,
        "gpr_count": shader_info.gpr_count,
        "workgroup": shader_info.workgroup,
        "shared_mem_bytes": shader_info.shared_mem_bytes,
        "barrier_count": shader_info.barrier_count,
        "buffer_data": buffer_data,
    });

    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return Err(BarracudaError::Device(
            "SovereignDevice: no tokio runtime available for IPC dispatch".into(),
        ));
    };

    let addr = dispatch_addr.to_owned();
    tokio::task::block_in_place(|| {
        handle.block_on(async {
            let host_port = addr.trim_start_matches("http://");
            let body = serde_json::to_string(&serde_json::json!({
                "jsonrpc": "2.0",
                "method": "compute.dispatch.submit",
                "params": [request],
                "id": 1,
            }))
            .map_err(|e| BarracudaError::Device(format!("serialize dispatch: {e}")))?;

            let http_request = format!(
                "POST / HTTP/1.1\r\nHost: {host_port}\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            );

            let mut stream = tokio::net::TcpStream::connect(host_port)
                .await
                .map_err(|e| {
                    BarracudaError::Device(format!(
                        "SovereignDevice: dispatch connect to {host_port}: {e}"
                    ))
                })?;

            tokio::io::AsyncWriteExt::write_all(&mut stream, http_request.as_bytes())
                .await
                .map_err(|e| {
                    BarracudaError::Device(format!("SovereignDevice: dispatch write: {e}"))
                })?;

            let mut response_buf = Vec::new();
            tokio::io::AsyncReadExt::read_to_end(&mut stream, &mut response_buf)
                .await
                .map_err(|e| {
                    BarracudaError::Device(format!("SovereignDevice: dispatch read: {e}"))
                })?;

            let response_str = String::from_utf8_lossy(&response_buf);
            let json_start = response_str.find('{').ok_or_else(|| {
                BarracudaError::Device("SovereignDevice: no JSON body in dispatch response".into())
            })?;

            let rpc_response: serde_json::Value = serde_json::from_str(&response_str[json_start..])
                .map_err(|e| {
                    BarracudaError::Device(format!("SovereignDevice: parse response: {e}"))
                })?;

            if let Some(error) = rpc_response.get("error") {
                let msg = error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                return Err(BarracudaError::Device(format!(
                    "SovereignDevice: compute dispatch error: {msg}"
                )));
            }

            if let Some(result) = rpc_response.get("result")
                && let Some(output_buffers) = result.get("output_buffers")
                && let Some(arr) = output_buffers.as_array()
            {
                let mut staged = staged_buffers.lock().map_err(|e| {
                    BarracudaError::Device(format!("SovereignDevice: staged lock poisoned: {e}"))
                })?;
                for entry in arr {
                    let Some(buf_id) = entry.get("buffer_id").and_then(serde_json::Value::as_u64)
                    else {
                        continue;
                    };
                    let Some(data) = entry.get("data").and_then(serde_json::Value::as_array) else {
                        continue;
                    };
                    let bytes: Vec<u8> = data
                        .iter()
                        .filter_map(|b| b.as_u64().map(|v| v as u8))
                        .collect();
                    if let Some(buf) = staged.get_mut(&buf_id) {
                        let copy_len = bytes.len().min(buf.len());
                        buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
                    }
                }
            }

            tracing::debug!(
                addr = %host_port,
                "sovereign dispatch submitted to compute.dispatch endpoint"
            );
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn test_bindings() -> Vec<IpcBufferBinding> {
        vec![IpcBufferBinding {
            index: 0,
            buffer_id: 42,
            size: 1024,
            read_only: false,
        }]
    }

    fn test_info() -> ShaderDispatchInfo {
        ShaderDispatchInfo {
            gpr_count: 32,
            workgroup: [64, 1, 1],
            shared_mem_bytes: 0,
            barrier_count: 0,
        }
    }

    fn multi_thread_rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn connection_refused_returns_device_error() {
        let rt = multi_thread_rt();
        let staged = Mutex::new(HashMap::new());
        let result = rt.block_on(async {
            submit_dispatch(
                "http://127.0.0.1:1",
                &staged,
                &[0xDE, 0xAD],
                (1, 1, 1),
                &test_bindings(),
                HardwareHint::Compute,
                test_info(),
            )
        });
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("dispatch connect") || msg.contains("Connection refused"),
            "Expected connection error, got: {msg}"
        );
    }

    #[test]
    fn malformed_response_returns_device_error() {
        let rt = multi_thread_rt();

        let listener =
            rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
        let addr = listener.local_addr().unwrap();

        rt.block_on(async {
            let server_task = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let garbage = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nnot-json!!!!";
                tokio::io::AsyncWriteExt::write_all(&mut stream, garbage)
                    .await
                    .unwrap();
                tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
            });

            let staged = Mutex::new(HashMap::new());
            let result = submit_dispatch(
                &format!("http://127.0.0.1:{}", addr.port()),
                &staged,
                &[0xDE, 0xAD],
                (1, 1, 1),
                &test_bindings(),
                HardwareHint::Compute,
                test_info(),
            );

            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("parse response") || msg.contains("no JSON body"),
                "Expected parse error, got: {msg}"
            );
            server_task.await.ok();
        });
    }

    #[test]
    fn rpc_error_in_response_returns_device_error() {
        let rt = multi_thread_rt();

        let listener =
            rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
        let addr = listener.local_addr().unwrap();

        rt.block_on(async {
            let server_task = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let body = r#"{"jsonrpc":"2.0","error":{"code":-32000,"message":"GPU memory exhausted"},"id":1}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                    .await
                    .unwrap();
                tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
            });

            let staged = Mutex::new(HashMap::new());
            let result = submit_dispatch(
                &format!("http://127.0.0.1:{}", addr.port()),
                &staged,
                &[0xDE, 0xAD],
                (1, 1, 1),
                &test_bindings(),
                HardwareHint::Compute,
                test_info(),
            );

            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("GPU memory exhausted"),
                "Expected RPC error message, got: {msg}"
            );
            server_task.await.ok();
        });
    }

    #[test]
    fn successful_dispatch_with_output_buffers() {
        let rt = multi_thread_rt();

        let listener =
            rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
        let addr = listener.local_addr().unwrap();

        rt.block_on(async {
            let server_task = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 8192];
                let n =
                    tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await.unwrap();
                let req_str = String::from_utf8_lossy(&buf[..n]);
                assert!(req_str.contains("compute.dispatch.submit"));

                let body = r#"{"jsonrpc":"2.0","result":{"output_buffers":[{"buffer_id":42,"data":[1,2,3,4]}]},"id":1}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                    .await
                    .unwrap();
                tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
            });

            let staged = Mutex::new(HashMap::new());
            staged
                .lock()
                .unwrap()
                .insert(42u64, bytes::BytesMut::from(&[0u8; 4][..]));

            let result = submit_dispatch(
                &format!("http://127.0.0.1:{}", addr.port()),
                &staged,
                &[0xDE, 0xAD],
                (1, 1, 1),
                &test_bindings(),
                HardwareHint::Compute,
                test_info(),
            );

            assert!(result.is_ok(), "dispatch should succeed: {:?}", result.err());
            {
                let locked = staged.lock().unwrap();
                let buf = locked.get(&42).unwrap();
                assert_eq!(&buf[..4], &[1, 2, 3, 4]);
            }
            server_task.await.ok();
        });
    }

    #[test]
    fn hardware_hint_serialized_correctly() {
        let hints = [
            (HardwareHint::Compute, "compute"),
            (HardwareHint::TensorCore, "tensor_core"),
            (HardwareHint::RtCore, "rt_core"),
            (HardwareHint::ZBuffer, "zbuffer"),
            (HardwareHint::TextureUnit, "texture_unit"),
            (HardwareHint::RopBlend, "rop_blend"),
        ];

        let rt = multi_thread_rt();

        for (hint, expected_str) in hints {
            let listener =
                rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
            let addr = listener.local_addr().unwrap();
            let expected_owned = expected_str.to_owned();

            rt.block_on(async {
                let server_task = tokio::spawn(async move {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let mut buf = vec![0u8; 16384];
                    let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                        .await
                        .unwrap();
                    let req_str = String::from_utf8_lossy(&buf[..n]);
                    assert!(
                        req_str.contains(&format!("\"hardware_hint\":\"{expected_owned}\"")),
                        "Expected hint '{expected_owned}' in request: {req_str}"
                    );
                    let body = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    );
                    tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                        .await
                        .unwrap();
                    tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
                });

                let staged = Mutex::new(HashMap::new());
                let result = submit_dispatch(
                    &format!("http://127.0.0.1:{}", addr.port()),
                    &staged,
                    &[0x00],
                    (1, 1, 1),
                    &test_bindings(),
                    hint,
                    test_info(),
                );
                assert!(
                    result.is_ok(),
                    "hint {expected_str} failed: {:?}",
                    result.err()
                );
                server_task.await.ok();
            });
        }
    }
}
