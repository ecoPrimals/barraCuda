// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binary serialization for `SimpleMlp` models.
//!
//! Implements the `BCML` (barraCuda ML) binary format with BLAKE3 integrity
//! verification and automatic format detection (binary vs JSON).

use super::SimpleMlp;

/// Magic bytes identifying a barraCuda ML binary model file.
pub(crate) const MODEL_MAGIC: &[u8; 4] = b"BCML";
/// Header version (1 = initial release).
const MODEL_VERSION: u8 = 1;
/// Format tag for bincode-encoded payload.
const FORMAT_BINCODE: u8 = 1;
/// Total header size: magic(4) + version(1) + format(1) + reserved(2) + len(4) + blake3(32).
pub(crate) const MODEL_HEADER_SIZE: usize = 44;

/// Errors from binary model serialization/deserialization.
#[derive(Debug, thiserror::Error)]
pub enum ModelBinaryError {
    /// File too short to contain a valid header.
    #[error("data too short ({0} bytes, need at least {MODEL_HEADER_SIZE})")]
    TooShort(usize),
    /// File does not start with `BCML` magic.
    #[error("invalid magic bytes (expected BCML header)")]
    BadMagic,
    /// Unsupported header version.
    #[error("unsupported model version {0} (expected {MODEL_VERSION})")]
    UnsupportedVersion(u8),
    /// Unsupported format tag.
    #[error("unsupported format tag {0}")]
    UnsupportedFormat(u8),
    /// BLAKE3 checksum mismatch — data corrupted or tampered.
    #[error("BLAKE3 checksum mismatch (data integrity failure)")]
    ChecksumMismatch,
    /// Payload exceeds u32 address space.
    #[error("payload too large ({0} bytes, max 4GB)")]
    PayloadTooLarge(usize),
    /// Encoding failed.
    #[error("encode error: {0}")]
    Encode(String),
    /// Decoding failed.
    #[error("decode error: {0}")]
    Decode(String),
}

impl SimpleMlp {
    /// Serialize to binary format with BLAKE3 integrity header.
    ///
    /// File layout (44-byte header + payload):
    /// - Magic: `BCML` (4 bytes)
    /// - Version: u8 (1 = initial)
    /// - Format: u8 (1 = bincode)
    /// - Reserved: 2 bytes
    /// - Payload length: u32 LE
    /// - BLAKE3 checksum of payload: 32 bytes
    /// - Payload (bincode-encoded `SimpleMlp`)
    /// # Errors
    /// Returns an error if bincode serialization fails.
    pub fn to_binary(&self) -> Result<Vec<u8>, ModelBinaryError> {
        let payload = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| ModelBinaryError::Encode(e.to_string()))?;

        let checksum = blake3::hash(&payload);
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| ModelBinaryError::PayloadTooLarge(payload.len()))?;

        let mut buf = Vec::with_capacity(MODEL_HEADER_SIZE + payload.len());
        buf.extend_from_slice(MODEL_MAGIC);
        buf.push(MODEL_VERSION);
        buf.push(FORMAT_BINCODE);
        buf.extend_from_slice(&[0u8; 2]);
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(checksum.as_bytes());
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    /// Deserialize from binary format, verifying BLAKE3 integrity.
    /// # Errors
    /// Returns an error if the header is invalid, checksum fails, or
    /// bincode deserialization fails.
    pub fn from_binary(data: &[u8]) -> Result<Self, ModelBinaryError> {
        if data.len() < MODEL_HEADER_SIZE {
            return Err(ModelBinaryError::TooShort(data.len()));
        }
        if &data[0..4] != MODEL_MAGIC {
            return Err(ModelBinaryError::BadMagic);
        }
        let version = data[4];
        if version != MODEL_VERSION {
            return Err(ModelBinaryError::UnsupportedVersion(version));
        }
        let format = data[5];
        if format != FORMAT_BINCODE {
            return Err(ModelBinaryError::UnsupportedFormat(format));
        }

        let payload_len =
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let expected_total = MODEL_HEADER_SIZE + payload_len;
        if data.len() < expected_total {
            return Err(ModelBinaryError::TooShort(data.len()));
        }

        let stored_checksum: [u8; 32] = data[12..44].try_into().unwrap_or([0u8; 32]);
        let payload = &data[MODEL_HEADER_SIZE..expected_total];
        let computed = blake3::hash(payload);
        if computed.as_bytes() != &stored_checksum {
            return Err(ModelBinaryError::ChecksumMismatch);
        }

        let (mlp, _) =
            bincode::serde::decode_from_slice::<Self, _>(payload, bincode::config::standard())
                .map_err(|e| ModelBinaryError::Decode(e.to_string()))?;
        Ok(mlp)
    }

    /// Detect format from file bytes and deserialize accordingly.
    ///
    /// If the file starts with the `BCML` magic header, deserializes as binary.
    /// Otherwise falls back to JSON parsing.
    /// # Errors
    /// Returns an error if neither format succeeds.
    pub fn from_auto(data: &[u8]) -> Result<Self, ModelBinaryError> {
        if data.len() >= 4 && &data[0..4] == MODEL_MAGIC {
            Self::from_binary(data)
        } else {
            let json_str = std::str::from_utf8(data)
                .map_err(|e| ModelBinaryError::Decode(e.to_string()))?;
            Self::from_json(json_str)
                .map_err(|e| ModelBinaryError::Decode(e.to_string()))
        }
    }
}
