// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP Phase 3 length-prefixed frame I/O with cipher support.
//!
//! Per `BTSP_PROTOCOL_STANDARD.md` §Wire Framing: all BTSP frames use a
//! 4-byte big-endian length prefix. Payload format depends on cipher suite:
//!
//! - `BTSP_NULL`: raw JSON-RPC plaintext
//! - `BTSP_HMAC_PLAIN`: plaintext ‖ hmac(32)
//! - `BTSP_CHACHA20_POLY1305`: nonce(12) ‖ ciphertext ‖ tag(16)
//!
//! Maximum frame size: 16 MiB (`0x0100_0000`).

use super::btsp::{BtspCipher, BtspSession};
use chacha20poly1305::{
    ChaCha20Poly1305, Nonce,
    aead::{Aead, KeyInit},
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const MAX_FRAME_SIZE: u32 = 0x0100_0000; // 16 MiB
const HMAC_TAG_LEN: usize = 32;
const CHACHA_NONCE_LEN: usize = 12;

type HmacSha256 = Hmac<Sha256>;

/// Errors from BTSP frame I/O operations.
#[derive(Debug, thiserror::Error)]
pub enum BtspFrameError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Frame exceeds the 16 MiB maximum.
    #[error("frame too large: {0} bytes (max {MAX_FRAME_SIZE})")]
    FrameTooLarge(u32),
    /// AEAD decryption or HMAC verification failed.
    #[error("authentication failed: {0}")]
    AuthFailed(String),
    /// Frame payload too short for the cipher's overhead.
    #[error("truncated frame: expected at least {expected} bytes, got {actual}")]
    Truncated {
        /// Minimum expected bytes.
        expected: usize,
        /// Actual bytes received.
        actual: usize,
    },
}

/// Reader for length-prefixed BTSP frames with decryption.
pub struct BtspFrameReader<R> {
    inner: R,
    cipher: BtspCipher,
    session_key: Vec<u8>,
    frame_counter: u64,
}

impl<R: AsyncRead + Unpin> BtspFrameReader<R> {
    /// Wrap a reader in a BTSP frame decoder.
    pub fn new(inner: R, session: &BtspSession) -> Self {
        Self {
            inner,
            cipher: session.cipher,
            session_key: session.session_key.clone(),
            frame_counter: 0,
        }
    }

    /// Read and decrypt the next frame, returning the plaintext payload.
    /// Returns `None` on clean EOF (peer closed).
    pub async fn read_frame(&mut self) -> Result<Option<Vec<u8>>, BtspFrameError> {
        let mut len_buf = [0u8; 4];
        match self.inner.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        let frame_len = u32::from_be_bytes(len_buf);
        if frame_len > MAX_FRAME_SIZE {
            return Err(BtspFrameError::FrameTooLarge(frame_len));
        }

        let mut payload = vec![0u8; frame_len as usize];
        self.inner.read_exact(&mut payload).await?;

        let plaintext = match self.cipher {
            BtspCipher::Null => payload,
            BtspCipher::HmacPlain => self.verify_hmac(payload)?,
            BtspCipher::ChaCha20Poly1305 => self.decrypt_chacha(payload)?,
        };

        self.frame_counter += 1;
        Ok(Some(plaintext))
    }

    fn verify_hmac(&self, payload: Vec<u8>) -> Result<Vec<u8>, BtspFrameError> {
        if payload.len() < HMAC_TAG_LEN {
            return Err(BtspFrameError::Truncated {
                expected: HMAC_TAG_LEN,
                actual: payload.len(),
            });
        }
        let split = payload.len() - HMAC_TAG_LEN;
        let (data, tag) = payload.split_at(split);

        let mut mac = <HmacSha256 as Mac>::new_from_slice(&self.session_key)
            .map_err(|e| BtspFrameError::AuthFailed(format!("HMAC key rejected: {e}")))?;
        mac.update(data);
        mac.verify_slice(tag)
            .map_err(|_| BtspFrameError::AuthFailed("HMAC-SHA256 verification failed".into()))?;

        Ok(data.to_vec())
    }

    fn decrypt_chacha(&self, payload: Vec<u8>) -> Result<Vec<u8>, BtspFrameError> {
        let min_len = CHACHA_NONCE_LEN + 16; // nonce + tag
        if payload.len() < min_len {
            return Err(BtspFrameError::Truncated {
                expected: min_len,
                actual: payload.len(),
            });
        }

        let (nonce_bytes, ciphertext) = payload.split_at(CHACHA_NONCE_LEN);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = ChaCha20Poly1305::new_from_slice(&self.session_key)
            .map_err(|e| BtspFrameError::AuthFailed(format!("invalid session key length: {e}")))?;

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| BtspFrameError::AuthFailed("ChaCha20-Poly1305 decryption failed".into()))
    }
}

/// Writer for length-prefixed BTSP frames with encryption.
pub struct BtspFrameWriter<W> {
    inner: W,
    cipher: BtspCipher,
    session_key: Vec<u8>,
    frame_counter: u64,
}

impl<W: AsyncWrite + Unpin> BtspFrameWriter<W> {
    /// Wrap a writer in a BTSP frame encoder.
    pub fn new(inner: W, session: &BtspSession) -> Self {
        Self {
            inner,
            cipher: session.cipher,
            session_key: session.session_key.clone(),
            frame_counter: 0,
        }
    }

    /// Encrypt and write a frame containing `plaintext`.
    pub async fn write_frame(&mut self, plaintext: &[u8]) -> Result<(), BtspFrameError> {
        let payload = match self.cipher {
            BtspCipher::Null => plaintext.to_vec(),
            BtspCipher::HmacPlain => self.compute_hmac(plaintext)?,
            BtspCipher::ChaCha20Poly1305 => self.encrypt_chacha(plaintext)?,
        };

        let len =
            u32::try_from(payload.len()).map_err(|_| BtspFrameError::FrameTooLarge(u32::MAX))?;
        if len > MAX_FRAME_SIZE {
            return Err(BtspFrameError::FrameTooLarge(len));
        }

        self.inner.write_all(&len.to_be_bytes()).await?;
        self.inner.write_all(&payload).await?;
        self.inner.flush().await?;

        self.frame_counter += 1;
        Ok(())
    }

    fn compute_hmac(&self, plaintext: &[u8]) -> Result<Vec<u8>, BtspFrameError> {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(&self.session_key)
            .map_err(|e| BtspFrameError::AuthFailed(format!("HMAC key rejected: {e}")))?;
        mac.update(plaintext);
        let tag = mac.finalize().into_bytes();

        let mut out = Vec::with_capacity(plaintext.len() + HMAC_TAG_LEN);
        out.extend_from_slice(plaintext);
        out.extend_from_slice(&tag);
        Ok(out)
    }

    fn encrypt_chacha(&self, plaintext: &[u8]) -> Result<Vec<u8>, BtspFrameError> {
        let cipher = ChaCha20Poly1305::new_from_slice(&self.session_key)
            .map_err(|e| BtspFrameError::AuthFailed(format!("invalid session key length: {e}")))?;

        // 12-byte nonce: 4 zero bytes ‖ 8-byte counter (per BTSP spec)
        let mut nonce_bytes = [0u8; CHACHA_NONCE_LEN];
        nonce_bytes[4..].copy_from_slice(&self.frame_counter.to_be_bytes());
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher.encrypt(nonce, plaintext).map_err(|_| {
            BtspFrameError::AuthFailed("ChaCha20-Poly1305 encryption failed".into())
        })?;

        let mut out = Vec::with_capacity(CHACHA_NONCE_LEN + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Flush and return the inner writer (useful for clean shutdown).
    pub async fn into_inner(mut self) -> Result<W, BtspFrameError> {
        self.inner.flush().await?;
        Ok(self.inner)
    }
}

/// Adaptor: read BTSP frames as newline-free JSON-RPC strings.
///
/// Bridges Phase 3 framed connections into the existing `handle_connection`
/// flow. Each frame is decoded and yielded as a `String` (the JSON-RPC
/// message). Returns `None` on EOF or auth failure.
pub async fn read_frame_as_line<R: AsyncRead + Unpin>(
    reader: &mut BtspFrameReader<R>,
) -> Option<String> {
    match reader.read_frame().await {
        Ok(Some(bytes)) => String::from_utf8(bytes).ok(),
        Ok(None) => None,
        Err(e) => {
            tracing::warn!("BTSP frame read error: {e}");
            None
        }
    }
}

/// Adaptor: write a JSON-RPC response string as a BTSP frame.
pub async fn write_line_as_frame<W: AsyncWrite + Unpin>(
    writer: &mut BtspFrameWriter<W>,
    json: &str,
) -> Result<(), BtspFrameError> {
    writer.write_frame(json.as_bytes()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn null_session() -> BtspSession {
        BtspSession {
            session_id: "test".into(),
            cipher: BtspCipher::Null,
            session_key: Vec::new(),
        }
    }

    fn hmac_session() -> BtspSession {
        BtspSession {
            session_id: "test-hmac".into(),
            cipher: BtspCipher::HmacPlain,
            session_key: vec![0xAB; 32],
        }
    }

    fn chacha_session() -> BtspSession {
        BtspSession {
            session_id: "test-chacha".into(),
            cipher: BtspCipher::ChaCha20Poly1305,
            session_key: vec![0x42; 32],
        }
    }

    #[tokio::test]
    async fn null_cipher_roundtrip() {
        let session = null_session();
        let msg = b"hello world";

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        writer.write_frame(msg).await.expect("write");

        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        let frame = reader.read_frame().await.expect("read").expect("not eof");
        assert_eq!(frame, msg);
    }

    #[tokio::test]
    async fn hmac_plain_roundtrip() {
        let session = hmac_session();
        let msg = br#"{"jsonrpc":"2.0","method":"health.check","id":1}"#;

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        writer.write_frame(msg).await.expect("write");

        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        let frame = reader.read_frame().await.expect("read").expect("not eof");
        assert_eq!(frame, msg);
    }

    #[tokio::test]
    async fn hmac_plain_tamper_detected() {
        let session = hmac_session();
        let msg = b"test payload";

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        writer.write_frame(msg).await.expect("write");

        // Tamper with a byte in the plaintext portion (after the 4-byte length header)
        if buf.len() > 5 {
            buf[4] ^= 0xFF;
        }

        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        let result = reader.read_frame().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn chacha20_roundtrip() {
        let session = chacha_session();
        let msg = br#"{"jsonrpc":"2.0","result":{"status":"alive"},"id":1}"#;

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        writer.write_frame(msg).await.expect("write");

        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        let frame = reader.read_frame().await.expect("read").expect("not eof");
        assert_eq!(frame, msg);
    }

    #[tokio::test]
    async fn chacha20_tamper_detected() {
        let session = chacha_session();
        let msg = b"secret payload";

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        writer.write_frame(msg).await.expect("write");

        // Tamper with a byte in the ciphertext (after 4-byte length + 12-byte nonce)
        if buf.len() > 17 {
            buf[17] ^= 0xFF;
        }

        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        let result = reader.read_frame().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn wrong_key_fails_chacha() {
        let session = chacha_session();
        let msg = b"confidential";

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        writer.write_frame(msg).await.expect("write");

        let wrong_session = BtspSession {
            session_id: "wrong".into(),
            cipher: BtspCipher::ChaCha20Poly1305,
            session_key: vec![0x99; 32],
        };
        let mut reader = BtspFrameReader::new(buf.as_slice(), &wrong_session);
        let result = reader.read_frame().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn wrong_key_fails_hmac() {
        let session = hmac_session();
        let msg = b"integrity-protected";

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        writer.write_frame(msg).await.expect("write");

        let wrong_session = BtspSession {
            session_id: "wrong".into(),
            cipher: BtspCipher::HmacPlain,
            session_key: vec![0x99; 32],
        };
        let mut reader = BtspFrameReader::new(buf.as_slice(), &wrong_session);
        let result = reader.read_frame().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn eof_returns_none() {
        let session = null_session();
        let buf: &[u8] = &[];
        let mut reader = BtspFrameReader::new(buf, &session);
        assert!(reader.read_frame().await.expect("no error").is_none());
    }

    #[tokio::test]
    async fn multiple_frames_roundtrip() {
        let session = chacha_session();
        let messages: Vec<&[u8]> = vec![b"frame-1", b"frame-2", b"frame-3"];

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        for msg in &messages {
            writer.write_frame(msg).await.expect("write");
        }

        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        for expected in &messages {
            let frame = reader.read_frame().await.expect("read").expect("not eof");
            assert_eq!(frame, *expected);
        }
        assert!(reader.read_frame().await.expect("no error").is_none());
    }

    #[tokio::test]
    async fn frame_too_large_rejected() {
        let session = null_session();
        let bad_len = (MAX_FRAME_SIZE + 1).to_be_bytes();
        let buf = bad_len.to_vec();
        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        let result = reader.read_frame().await;
        assert!(matches!(result, Err(BtspFrameError::FrameTooLarge(_))));
    }

    #[tokio::test]
    async fn line_adaptors_roundtrip() {
        let session = null_session();
        let json = r#"{"jsonrpc":"2.0","method":"health.check","id":1}"#;

        let mut buf = Vec::new();
        let mut writer = BtspFrameWriter::new(&mut buf, &session);
        write_line_as_frame(&mut writer, json).await.expect("write");

        let mut reader = BtspFrameReader::new(buf.as_slice(), &session);
        let line = read_frame_as_line(&mut reader).await.expect("not eof");
        assert_eq!(line, json);
    }
}
