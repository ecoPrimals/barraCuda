// SPDX-License-Identifier: AGPL-3.0-or-later
//! Genetics-layer prefix stripping: mito-beacon v1/v2, nuclear lineage,
//! plain JSON pass-through, edge cases (empty stream, single byte).

use super::*;

#[tokio::test]
async fn genetics_prefix_mito_v1_stripped_before_json() {
    use tokio::io::BufReader;

    let payload = b"\xEC\x01{\"jsonrpc\":\"2.0\",\"method\":\"health\",\"id\":1}\n";
    let cursor = std::io::Cursor::new(payload.to_vec());
    let mut reader = BufReader::new(cursor);

    let stripped = strip_genetics_prefix(&mut reader).await;
    assert!(
        stripped,
        "mito-beacon v1 prefix should be detected and stripped"
    );

    let mut line = String::new();
    tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut line)
        .await
        .unwrap();
    assert_eq!(line.trim(), r#"{"jsonrpc":"2.0","method":"health","id":1}"#);
}

#[tokio::test]
async fn genetics_prefix_mito_v2_stripped() {
    use tokio::io::BufReader;

    let payload = b"\xED\x01{\"jsonrpc\":\"2.0\",\"method\":\"health\",\"id\":1}\n";
    let cursor = std::io::Cursor::new(payload.to_vec());
    let mut reader = BufReader::new(cursor);

    let stripped = strip_genetics_prefix(&mut reader).await;
    assert!(
        stripped,
        "mito-beacon v2 prefix should be detected and stripped"
    );

    let mut line = String::new();
    tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut line)
        .await
        .unwrap();
    assert_eq!(line.trim(), r#"{"jsonrpc":"2.0","method":"health","id":1}"#);
}

#[tokio::test]
async fn genetics_prefix_nuclear_lineage_stripped() {
    use tokio::io::BufReader;

    let payload = b"\xEE\x02{\"jsonrpc\":\"2.0\",\"method\":\"health\",\"id\":1}\n";
    let cursor = std::io::Cursor::new(payload.to_vec());
    let mut reader = BufReader::new(cursor);

    let stripped = strip_genetics_prefix(&mut reader).await;
    assert!(
        stripped,
        "nuclear lineage prefix should be detected and stripped"
    );

    let mut line = String::new();
    tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut line)
        .await
        .unwrap();
    assert_eq!(line.trim(), r#"{"jsonrpc":"2.0","method":"health","id":1}"#);
}

#[tokio::test]
async fn genetics_prefix_not_stripped_from_plain_json() {
    use tokio::io::BufReader;

    let payload = b"{\"jsonrpc\":\"2.0\",\"method\":\"health\",\"id\":1}\n";
    let cursor = std::io::Cursor::new(payload.to_vec());
    let mut reader = BufReader::new(cursor);

    let stripped = strip_genetics_prefix(&mut reader).await;
    assert!(!stripped, "plain JSON should not trigger prefix stripping");

    let mut line = String::new();
    tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut line)
        .await
        .unwrap();
    assert_eq!(line.trim(), r#"{"jsonrpc":"2.0","method":"health","id":1}"#);
}

#[tokio::test]
async fn genetics_prefix_empty_stream_no_panic() {
    use tokio::io::BufReader;

    let cursor = std::io::Cursor::new(Vec::<u8>::new());
    let mut reader = BufReader::new(cursor);

    let stripped = strip_genetics_prefix(&mut reader).await;
    assert!(!stripped);
}

#[tokio::test]
async fn genetics_prefix_single_byte_no_consume() {
    use tokio::io::BufReader;

    let payload = b"\xEC";
    let cursor = std::io::Cursor::new(payload.to_vec());
    let mut reader = BufReader::new(cursor);

    let stripped = strip_genetics_prefix(&mut reader).await;
    assert!(
        !stripped,
        "single byte should not be consumed (need 2-byte prefix)"
    );
}
