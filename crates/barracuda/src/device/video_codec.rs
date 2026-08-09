// SPDX-License-Identifier: AGPL-3.0-or-later
//! Abstraction over hardware video encoding/decoding.
//!
//! Backends: NVENC (NVIDIA), VAAPI (AMD/Intel), software fallback.
//! Detection is capability-based — if ffmpeg with the right backend is
//! available on the system, we use it. Otherwise graceful degradation.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;
use std::process::Command;

/// Environment variable override for the ffmpeg binary path.
pub const VIDEO_CODEC_PATH_ENV: &str = "BARRACUDA_VIDEO_CODEC_PATH";

/// Abstraction over hardware video encoding/decoding.
///
/// Backends: NVENC (NVIDIA), VAAPI (AMD/Intel), software fallback.
/// Detection is capability-based — if ffmpeg with the right backend is
/// available on the system, we use it. Otherwise graceful degradation.
pub trait VideoCodec: Send + Sync {
    /// Encode raw frames to compressed video bytes.
    ///
    /// # Errors
    /// Returns `CodecError` if encoding fails or is not available.
    fn encode(&self, frames: &[FrameData], config: &EncodeConfig) -> Result<Vec<u8>, CodecError>;
    /// Decode compressed video bytes to raw frames.
    ///
    /// # Errors
    /// Returns `CodecError` if decoding fails or is not available.
    fn decode(&self, compressed: &[u8], config: &DecodeConfig) -> Result<Vec<FrameData>, CodecError>;
    /// Name of this codec backend.
    fn backend_name(&self) -> &str;
    /// Estimated compression ratio (e.g. 61.0 for NVENC on lattice data).
    fn estimated_compression_ratio(&self) -> f64;
}

/// Raw video frame payload for encode/decode.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FrameData {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Raw pixel data (layout depends on `format`).
    pub data: Vec<u8>,
    /// Pixel format of `data`.
    pub format: PixelFormat,
}

/// Pixel layout for frame buffers.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PixelFormat {
    /// 4-channel 8-bit RGBA.
    Rgba8,
    /// 3-channel 8-bit RGB.
    Rgb8,
    /// Single-channel 8-bit grayscale.
    Gray8,
    /// Single-channel 32-bit float grayscale.
    Gray32f,
}

/// Encoder configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncodeConfig {
    /// Target compressed codec.
    pub codec: CodecType,
    /// Quality vs speed tradeoff.
    pub quality: Quality,
    /// Frames between keyframes (GOP size).
    pub keyframe_interval: u32,
}

/// Decoder configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeConfig {
    /// Expected pixel format of decoded frames.
    pub expected_format: PixelFormat,
}

/// Supported compressed video codecs.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CodecType {
    /// H.264 / AVC (NVENC, VAAPI, libx264).
    H264,
    /// H.265 / HEVC (NVENC, VAAPI, libx265).
    H265,
    /// AV1 (NVENC, VAAPI, libaom/libsvtav1).
    Av1,
}

/// Encoder quality preset.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Quality {
    /// Prioritize encoding speed over quality.
    Fast,
    /// Balanced encoding speed and quality.
    Balanced,
    /// Prioritize quality over encoding speed.
    Quality,
}

/// Codec operation failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// No suitable codec backend detected on this system.
    NotAvailable(String),
    /// Encoding failed.
    EncodeFailed(String),
    /// Decoding failed.
    DecodeFailed(String),
    /// Invalid input data or configuration.
    InvalidInput(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAvailable(msg) => write!(f, "codec not available: {msg}"),
            Self::EncodeFailed(msg) => write!(f, "encode failed: {msg}"),
            Self::DecodeFailed(msg) => write!(f, "decode failed: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Detected codec backend and supported compressed formats.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodecInfo {
    /// Backend name (e.g. "nvenc", "vaapi", "software").
    pub backend: String,
    /// Whether this backend uses hardware acceleration.
    pub hw_accel: bool,
    /// Codec types supported by this backend.
    pub codecs: Vec<CodecType>,
}

/// Placeholder codec used when no encoder is detected.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullCodec;

impl VideoCodec for NullCodec {
    fn encode(&self, _frames: &[FrameData], _config: &EncodeConfig) -> Result<Vec<u8>, CodecError> {
        Err(CodecError::NotAvailable(
            "no video codec backend available".into(),
        ))
    }

    fn decode(
        &self,
        _compressed: &[u8],
        _config: &DecodeConfig,
    ) -> Result<Vec<FrameData>, CodecError> {
        Err(CodecError::NotAvailable(
            "no video codec backend available".into(),
        ))
    }

    #[expect(clippy::unnecessary_literal_bound, reason = "trait constrains return lifetime")]
    fn backend_name(&self) -> &str {
        "null"
    }

    fn estimated_compression_ratio(&self) -> f64 {
        1.0
    }
}

/// Probe the system for available video codec backends.
///
/// Checks `BARRACUDA_VIDEO_CODEC_PATH` first, then `which ffmpeg` on PATH.
/// When ffmpeg is found, inspects its encoder list for NVENC, VAAPI, and
/// software fallbacks. Returns an empty vector when ffmpeg is unavailable.
#[must_use]
pub fn detect_codecs() -> Vec<CodecInfo> {
    let Some(ffmpeg) = resolve_ffmpeg_path() else {
        return Vec::new();
    };

    probe_ffmpeg_encoders(&ffmpeg)
}

/// Resolve the ffmpeg binary: custom env path, then PATH lookup.
fn resolve_ffmpeg_path() -> Option<String> {
    if let Ok(custom) = std::env::var(VIDEO_CODEC_PATH_ENV) {
        let trimmed = custom.trim();
        if !trimmed.is_empty() && Path::new(trimmed).is_file() {
            return Some(trimmed.to_owned());
        }
    }

    which_ffmpeg()
}

fn which_ffmpeg() -> Option<String> {
    let output = Command::new("which").arg("ffmpeg").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

fn probe_ffmpeg_encoders(ffmpeg: &str) -> Vec<CodecInfo> {
    let output = match Command::new(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => {
            return vec![CodecInfo {
                backend: "ffmpeg".into(),
                hw_accel: false,
                codecs: vec![CodecType::H264],
            }];
        }
    };

    let listing = String::from_utf8_lossy(&output.stdout);
    let mut codecs = Vec::new();

    if listing.contains("h264_nvenc") || listing.contains("hevc_nvenc") || listing.contains("av1_nvenc")
    {
        let mut nvenc_codecs = Vec::new();
        if listing.contains("h264_nvenc") {
            nvenc_codecs.push(CodecType::H264);
        }
        if listing.contains("hevc_nvenc") {
            nvenc_codecs.push(CodecType::H265);
        }
        if listing.contains("av1_nvenc") {
            nvenc_codecs.push(CodecType::Av1);
        }
        if !nvenc_codecs.is_empty() {
            codecs.push(CodecInfo {
                backend: "nvenc".into(),
                hw_accel: true,
                codecs: nvenc_codecs,
            });
        }
    }

    if listing.contains("h264_vaapi") || listing.contains("hevc_vaapi") || listing.contains("av1_vaapi")
    {
        let mut vaapi_codecs = Vec::new();
        if listing.contains("h264_vaapi") {
            vaapi_codecs.push(CodecType::H264);
        }
        if listing.contains("hevc_vaapi") {
            vaapi_codecs.push(CodecType::H265);
        }
        if listing.contains("av1_vaapi") {
            vaapi_codecs.push(CodecType::Av1);
        }
        if !vaapi_codecs.is_empty() {
            codecs.push(CodecInfo {
                backend: "vaapi".into(),
                hw_accel: true,
                codecs: vaapi_codecs,
            });
        }
    }

    let mut sw_codecs = Vec::new();
    if listing.contains("libx264") {
        sw_codecs.push(CodecType::H264);
    }
    if listing.contains("libx265") {
        sw_codecs.push(CodecType::H265);
    }
    if listing.contains("libaom-av1") || listing.contains("libsvtav1") {
        sw_codecs.push(CodecType::Av1);
    }
    if !sw_codecs.is_empty() {
        codecs.push(CodecInfo {
            backend: "software".into(),
            hw_accel: false,
            codecs: sw_codecs,
        });
    }

    if codecs.is_empty() {
        codecs.push(CodecInfo {
            backend: "ffmpeg".into(),
            hw_accel: false,
            codecs: vec![CodecType::H264],
        });
    }

    codecs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_codec_encode_returns_not_available() {
        let codec = NullCodec;
        let frame = FrameData {
            width: 64,
            height: 64,
            data: vec![0; 64 * 64 * 4],
            format: PixelFormat::Rgba8,
        };
        let config = EncodeConfig {
            codec: CodecType::H264,
            quality: Quality::Balanced,
            keyframe_interval: 30,
        };

        let err = codec.encode(&[frame], &config).unwrap_err();
        assert!(matches!(err, CodecError::NotAvailable(_)));
    }

    #[test]
    fn null_codec_decode_returns_not_available() {
        let codec = NullCodec;
        let config = DecodeConfig {
            expected_format: PixelFormat::Rgba8,
        };

        let err = codec.decode(&[0u8; 16], &config).unwrap_err();
        assert!(matches!(err, CodecError::NotAvailable(_)));
    }

    #[test]
    fn detect_codecs_does_not_panic() {
        let _codecs = detect_codecs();
    }

    #[test]
    fn codec_error_display_formatting() {
        assert_eq!(
            CodecError::NotAvailable("no ffmpeg".into()).to_string(),
            "codec not available: no ffmpeg"
        );
        assert_eq!(
            CodecError::EncodeFailed("timeout".into()).to_string(),
            "encode failed: timeout"
        );
        assert_eq!(
            CodecError::DecodeFailed("corrupt".into()).to_string(),
            "decode failed: corrupt"
        );
        assert_eq!(
            CodecError::InvalidInput("empty frames".into()).to_string(),
            "invalid input: empty frames"
        );
    }

    #[test]
    fn frame_data_creation() {
        let frame = FrameData {
            width: 1920,
            height: 1080,
            data: vec![128; 1920 * 1080 * 4],
            format: PixelFormat::Rgba8,
        };

        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.data.len(), 1920 * 1080 * 4);
        assert_eq!(frame.format, PixelFormat::Rgba8);
    }
}
