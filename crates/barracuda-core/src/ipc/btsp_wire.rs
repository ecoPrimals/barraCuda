// SPDX-License-Identifier: AGPL-3.0-or-later
//! NDJSON wire helpers for BTSP handshake and discovery communication.
//!
//! Newline-Delimited JSON (NDJSON) is the framing format per
//! `PRIMAL_IPC_PROTOCOL.md` v3.1. These helpers are shared between
//! `btsp.rs` (handshake relay) and `btsp_discovery.rs`.

/// Read a single NDJSON line from an async buffered reader.
///
/// Returns `UnexpectedEof` if the connection closes before a complete line.
pub(super) async fn read_ndjson_line<R>(reader: &mut R) -> std::io::Result<String>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    use tokio::io::AsyncBufReadExt;
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    if line.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "connection closed before NDJSON line",
        ));
    }
    Ok(line)
}

/// Write a JSON value as a single NDJSON line to an async writer.
///
/// Serializes the value, appends `\n`, writes, and flushes.
pub(super) async fn write_ndjson_line<W>(
    writer: &mut W,
    value: &serde_json::Value,
) -> std::io::Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;
    let mut line = serde_json::to_string(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}
