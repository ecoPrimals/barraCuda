// SPDX-License-Identifier: AGPL-3.0-or-later
use super::super::jsonrpc::JsonRpcResponse;
use super::dispatch::{self, SERIALIZATION_ERROR};
use crate::BarraCudaPrimal;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

/// Genetics-layer signal prefixes per eukaryotic model.
///
/// cellMembrane/NUCLEUS probers send a 2-byte prefix before JSON-RPC payloads
/// to distinguish live primals from stale sockets and signal transport intent.
/// Primals strip the prefix and continue with normal JSON-RPC parsing.
///
/// | Byte 0 | Stream              | Purpose                              |
/// |--------|---------------------|--------------------------------------|
/// | `0xEC` | MitoBeacon v1       | Relay access, mesh, ABG transport    |
/// | `0xED` | MitoBeacon v2       | Extended mito-beacon (future)        |
/// | `0xEE` | Nuclear Lineage     | Per-user permissions (BearDog-spawned)|
///
/// Byte 1 is a version/sub-type discriminator (currently `0x01`).
const MITO_BEACON_V1: u8 = 0xEC;
const MITO_BEACON_V2: u8 = 0xED;
const NUCLEAR_LINEAGE: u8 = 0xEE;

/// Strip a genetics-layer signal prefix from a BufReader if present.
///
/// Uses `fill_buf()` to peek at buffered bytes without consuming. If the
/// first byte matches a known genetics signal (`0xEC`, `0xED`, `0xEE`),
/// consumes the 2-byte prefix (signal byte + version). Otherwise the stream
/// remains untouched for normal JSON-RPC parsing. Works for both TCP and UDS
/// transports without platform-specific peek syscalls.
pub(super) async fn strip_genetics_prefix<R: AsyncRead + Unpin>(reader: &mut BufReader<R>) -> bool {
    match reader.fill_buf().await {
        Ok(buf)
            if buf.len() >= 2
                && matches!(buf[0], MITO_BEACON_V1 | MITO_BEACON_V2 | NUCLEAR_LINEAGE) =>
        {
            let signal = buf[0];
            reader.consume(2);
            tracing::debug!(
                signal = format_args!("0x{signal:02X}"),
                "genetics prefix accepted"
            );
            true
        }
        _ => false,
    }
}

/// Transport-agnostic JSON-RPC 2.0 connection handler.
///
/// Works over any `AsyncRead + AsyncWrite` stream — TCP, Unix socket,
/// or future transports (named pipes, abstract sockets). This is the
/// ecoBin v2.0 transport-agnostic protocol handler: add a new transport
/// by binding a listener and passing accepted connections here.
///
/// Supports both single requests and JSON-RPC 2.0 batch requests (JSON
/// arrays). Per spec: an empty batch returns a parse error; notifications
/// within a batch produce no response entries; if all requests are
/// notifications, no response is sent.
///
/// `replay` is an optional first line consumed during the BTSP handshake
/// guard. When `FAMILY_ID` is set and a legacy (non-BTSP) client connects,
/// the guard reads the first line looking for a `ClientHello`. If the line
/// is a plain JSON-RPC request, the guard returns it here for replay so
/// the request is not silently dropped (LD-10 fix).
pub(super) async fn handle_connection<R, W>(
    primal: Arc<BarraCudaPrimal>,
    reader: R,
    mut writer: W,
    session: Option<super::super::btsp::BtspSession>,
    replay: Option<String>,
) where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    if let Some(ref line) = replay
        && dispatch_line(&primal, line, &mut writer).await.is_err()
    {
        return;
    }
    let mut buf_reader = BufReader::new(reader);
    strip_genetics_prefix(&mut buf_reader).await;
    loop {
        let line = {
            let mut lines = (&mut buf_reader).lines();
            match lines.next_line().await {
                Ok(Some(l)) => l,
                _ => break,
            }
        };
        if let Some(upgraded) = try_negotiate_upgrade(session.as_ref(), &line, &mut writer).await {
            tracing::info!("Phase 3 negotiate: switching to encrypted framing");
            handle_btsp_connection(primal, buf_reader, writer, &upgraded).await;
            return;
        }
        if dispatch_line(&primal, &line, &mut writer).await.is_err() {
            break;
        }
    }
}

/// If `line` is a `btsp.negotiate` request and the session supports it,
/// write the response and return the upgraded `BtspSession`. Returns `None`
/// for non-negotiate lines or when the cipher stays NULL.
async fn try_negotiate_upgrade<W>(
    session: Option<&super::super::btsp::BtspSession>,
    line: &str,
    writer: &mut W,
) -> Option<super::super::btsp::BtspSession>
where
    W: AsyncWrite + Unpin + Send,
{
    let session = session.as_ref()?;
    let parsed: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    if parsed.get("method")?.as_str()? != "btsp.negotiate" {
        return None;
    }
    let id = parsed.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let params = parsed
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let (resp, upgraded) = match super::super::btsp::negotiate_phase3(session, &params) {
        Ok(nr) => {
            let keyed = nr.session.cipher.requires_key();
            (
                JsonRpcResponse::success(id, nr.response),
                keyed.then_some(nr.session),
            )
        }
        Err(e) => (JsonRpcResponse::error(id, -32602, e.to_string()), None),
    };
    let mut json = serde_json::to_string(&resp).unwrap_or_else(|_| SERIALIZATION_ERROR.to_string());
    json.push('\n');
    if writer.write_all(json.as_bytes()).await.is_err() {
        return None;
    }
    if upgraded.is_some() {
        let _ = writer.flush().await;
    }
    upgraded
}

/// Dispatch a single NDJSON line (single request or batch array) and write
/// the response. Returns `Err(())` on write failure (caller should close).
async fn dispatch_line<W>(
    primal: &Arc<BarraCudaPrimal>,
    line: &str,
    writer: &mut W,
) -> std::result::Result<(), ()>
where
    W: AsyncWrite + Unpin + Send,
{
    let trimmed = line.trim_start();
    if trimmed.starts_with('[') {
        let Some(batch_json) = dispatch::handle_batch(primal, trimmed).await else {
            return Ok(());
        };
        let mut out = batch_json;
        out.push('\n');
        writer.write_all(out.as_bytes()).await.map_err(|e| {
            tracing::debug!("IPC write failed (batch response): {e}");
        })?;
    } else {
        let Some(response) = dispatch::handle_line(primal, line).await else {
            return Ok(());
        };
        let mut json =
            serde_json::to_string(&response).unwrap_or_else(|_| SERIALIZATION_ERROR.to_string());
        json.push('\n');
        writer.write_all(json.as_bytes()).await.map_err(|e| {
            tracing::debug!("IPC write failed (single response): {e}");
        })?;
    }
    Ok(())
}

/// Handle a BTSP Phase 3 connection using length-prefixed encrypted frames.
///
/// Each frame is decrypted, dispatched as JSON-RPC, and the response
/// encrypted back. Per `BTSP_PROTOCOL_STANDARD.md` §Wire Framing.
pub(super) async fn handle_btsp_connection<R, W>(
    primal: Arc<BarraCudaPrimal>,
    reader: R,
    writer: W,
    session: &super::super::btsp::BtspSession,
) where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    use super::super::btsp_frame::{
        BtspFrameReader, BtspFrameWriter, read_frame_as_line, write_line_as_frame,
    };

    let mut frame_reader = BtspFrameReader::new(reader, session);
    let mut frame_writer = BtspFrameWriter::new(writer, session);

    while let Some(line) = read_frame_as_line(&mut frame_reader).await {
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') {
            let Some(batch_json) = dispatch::handle_batch(&primal, trimmed).await else {
                continue;
            };
            if write_line_as_frame(&mut frame_writer, &batch_json)
                .await
                .is_err()
            {
                break;
            }
        } else {
            let Some(response) = dispatch::handle_line(&primal, &line).await else {
                continue;
            };
            let json = serde_json::to_string(&response)
                .unwrap_or_else(|_| SERIALIZATION_ERROR.to_string());
            if write_line_as_frame(&mut frame_writer, &json).await.is_err() {
                break;
            }
        }
    }
}
