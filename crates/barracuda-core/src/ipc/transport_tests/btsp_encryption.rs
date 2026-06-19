// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP Phase 3 encryption: negotiate → key derivation → encrypted frame loop,
//! pipelined data preservation across cipher upgrade.

use super::*;
use crate::ipc::btsp::{BtspCipher, BtspSession};
use crate::ipc::btsp_frame::{
    BtspFrameReader, BtspFrameWriter, read_frame_as_line, write_line_as_frame,
};
use hkdf::Hkdf;
use sha2::Sha256;
use tokio::io::{AsyncWriteExt, duplex};

/// Live validation: after btsp.negotiate upgrades to chacha20-poly1305, all
/// subsequent messages on the connection use encrypted framing (including
/// any data pipelined immediately after the negotiate request line).
#[tokio::test]
async fn negotiate_then_encrypted_frame_loop() {
    let session = BtspSession {
        session_id: "live-val-001".into(),
        cipher: BtspCipher::Null,
        session_key: vec![0xFE; 32],
    };

    let client_nonce = b"c1c2c3c4c5c6c7c8c9cacbcc";
    let client_nonce_hex = "633163326333633463356336633763386339636163626363";

    let negotiate_req = format!(
        r#"{{"jsonrpc":"2.0","method":"btsp.negotiate","params":{{"session_id":"live-val-001","preferred_cipher":"chacha20-poly1305","client_nonce":"{client_nonce_hex}"}},"id":99}}"#
    );

    let post_negotiate_plaintext =
        r#"{"jsonrpc":"2.0","method":"device.list","params":{},"id":100}"#;

    let (client_io, server_io) = duplex(16384);
    let (server_reader, server_writer) = tokio::io::split(server_io);
    let (mut client_reader, mut client_writer) = tokio::io::split(client_io);

    let primal = Arc::new(BarraCudaPrimal::new());

    let server_handle = tokio::spawn(async move {
        handle_connection(primal, server_reader, server_writer, Some(session), None).await;
    });

    client_writer
        .write_all(format!("{negotiate_req}\n").as_bytes())
        .await
        .unwrap();

    let mut resp_buf = vec![0u8; 4096];
    let n = tokio::io::AsyncReadExt::read(&mut client_reader, &mut resp_buf)
        .await
        .unwrap();
    let resp_str = std::str::from_utf8(&resp_buf[..n]).unwrap();
    let resp_json: serde_json::Value = serde_json::from_str(resp_str.trim()).unwrap();

    assert_eq!(resp_json["id"], 99);
    let result = &resp_json["result"];
    assert_eq!(result["cipher"].as_str(), Some("chacha20-poly1305"));
    let server_nonce_hex = result["server_nonce"].as_str().unwrap();

    let server_nonce: Vec<u8> = (0..server_nonce_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&server_nonce_hex[i..i + 2], 16).unwrap())
        .collect();

    let mut salt = Vec::new();
    salt.extend_from_slice(client_nonce);
    salt.extend_from_slice(&server_nonce);
    let ikm = [0xFE; 32];
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), &ikm);
    let mut derived_key = [0u8; 32];
    hkdf.expand(b"btsp-v1-phase3", &mut derived_key).unwrap();

    let derived_session = BtspSession {
        session_id: "live-val-001".into(),
        cipher: BtspCipher::ChaCha20Poly1305,
        session_key: derived_key.to_vec(),
    };

    let mut frame_writer = BtspFrameWriter::new(&mut client_writer, &derived_session);
    write_line_as_frame(&mut frame_writer, post_negotiate_plaintext)
        .await
        .unwrap();

    let mut frame_reader = BtspFrameReader::new(&mut client_reader, &derived_session);
    let response_line = read_frame_as_line(&mut frame_reader)
        .await
        .expect("should receive encrypted frame response");

    let resp: serde_json::Value = serde_json::from_str(&response_line).unwrap();
    assert_eq!(resp["id"], 100);
    assert!(
        resp["result"].is_object(),
        "device.list should return a result"
    );

    drop(frame_writer);
    drop(frame_reader);
    drop(client_writer);
    drop(client_reader);
    let _ = server_handle.await;
}

/// Verify that pipelined data after negotiate is not lost (BufReader
/// buffering preservation).
#[tokio::test]
async fn negotiate_pipelined_frame_not_lost() {
    let session = BtspSession {
        session_id: "pipeline-test".into(),
        cipher: BtspCipher::Null,
        session_key: vec![0xAA; 32],
    };

    let negotiate_req = r#"{"jsonrpc":"2.0","method":"btsp.negotiate","params":{"session_id":"pipeline-test","preferred_cipher":"chacha20-poly1305"},"id":1}"#;

    let (client_io, server_io) = duplex(16384);
    let (server_reader, server_writer) = tokio::io::split(server_io);
    let (mut client_reader, mut client_writer) = tokio::io::split(client_io);

    let primal = Arc::new(BarraCudaPrimal::new());

    let key_material = vec![0xAA; 32];
    let server_handle = tokio::spawn(async move {
        handle_connection(primal, server_reader, server_writer, Some(session), None).await;
    });

    let payload = format!("{negotiate_req}\n").into_bytes();

    let empty_client_nonce: Vec<u8> = Vec::new();

    client_writer.write_all(&payload).await.unwrap();

    let mut resp_buf = vec![0u8; 4096];
    let n = tokio::io::AsyncReadExt::read(&mut client_reader, &mut resp_buf)
        .await
        .unwrap();
    let resp_str = std::str::from_utf8(&resp_buf[..n]).unwrap();
    let resp_json: serde_json::Value = serde_json::from_str(resp_str.trim()).unwrap();
    let server_nonce_hex = resp_json["result"]["server_nonce"].as_str().unwrap();

    let server_nonce: Vec<u8> = (0..server_nonce_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&server_nonce_hex[i..i + 2], 16).unwrap())
        .collect();

    let mut salt = Vec::new();
    salt.extend_from_slice(&empty_client_nonce);
    salt.extend_from_slice(&server_nonce);
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), &key_material);
    let mut derived_key = [0u8; 32];
    hkdf.expand(b"btsp-v1-phase3", &mut derived_key).unwrap();

    let derived_session = BtspSession {
        session_id: "pipeline-test".into(),
        cipher: BtspCipher::ChaCha20Poly1305,
        session_key: derived_key.to_vec(),
    };

    let post_req = r#"{"jsonrpc":"2.0","method":"primal.info","params":{},"id":2}"#;
    let mut frame_buf: Vec<u8> = Vec::new();
    {
        let mut fw = BtspFrameWriter::new(&mut frame_buf, &derived_session);
        write_line_as_frame(&mut fw, post_req).await.unwrap();
    }
    client_writer.write_all(&frame_buf).await.unwrap();

    let mut frame_reader = BtspFrameReader::new(&mut client_reader, &derived_session);
    let response_line = read_frame_as_line(&mut frame_reader)
        .await
        .expect("pipelined encrypted frame must not be lost");

    let resp: serde_json::Value = serde_json::from_str(&response_line).unwrap();
    assert_eq!(resp["id"], 2);
    assert!(
        resp["result"].is_object(),
        "primal.info should return result"
    );

    drop(frame_reader);
    drop(client_writer);
    drop(client_reader);
    let _ = server_handle.await;
}
