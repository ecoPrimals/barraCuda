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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let value = serde_json::json!({"method": "health.liveness", "id": 1});
        let mut buf = Vec::new();
        write_ndjson_line(&mut buf, &value).await.unwrap();

        let written = String::from_utf8(buf.clone()).unwrap();
        assert!(written.ends_with('\n'));
        assert_eq!(written.matches('\n').count(), 1);

        let mut reader = tokio::io::BufReader::new(buf.as_slice());
        let line = read_ndjson_line(&mut reader).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed, value);
    }

    #[tokio::test]
    async fn read_empty_stream_returns_eof() {
        let empty: &[u8] = &[];
        let mut reader = tokio::io::BufReader::new(empty);
        let err = read_ndjson_line(&mut reader).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn write_complex_value() {
        let value = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "compute.dispatch.submit",
            "params": {"binary_b64": "AQIDBA==", "bindings": [0, 1]},
            "id": 42
        });
        let mut buf = Vec::new();
        write_ndjson_line(&mut buf, &value).await.unwrap();

        let written = String::from_utf8(buf).unwrap();
        assert!(written.contains("compute.dispatch.submit"));
        assert!(written.ends_with('\n'));
        assert!(!written[..written.len() - 1].contains('\n'));
    }

    #[tokio::test]
    async fn multiple_lines_roundtrip() {
        let values = vec![
            serde_json::json!({"id": 1}),
            serde_json::json!({"id": 2}),
            serde_json::json!({"id": 3}),
        ];

        let mut buf = Vec::new();
        for v in &values {
            write_ndjson_line(&mut buf, v).await.unwrap();
        }

        let mut reader = tokio::io::BufReader::new(buf.as_slice());
        for expected in &values {
            let line = read_ndjson_line(&mut reader).await.unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(&parsed, expected);
        }
    }
}
