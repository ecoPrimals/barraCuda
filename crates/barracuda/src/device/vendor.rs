//! Canonical GPU vendor ID constants.
//!
//! Every place in BarraCuda that branches on `caps.vendor` or
//! `adapter_info.vendor` should import from here instead of
//! repeating raw hex literals.  This is the single source of truth.

/// NVIDIA Corporation (Green Team)
pub const VENDOR_NVIDIA: u32 = 0x10DE;

/// AMD (Radeon / RDNA)
pub const VENDOR_AMD: u32 = 0x1002;

/// Intel Corporation (Arc / Iris)
pub const VENDOR_INTEL: u32 = 0x8086;

/// Apple Silicon GPU (Metal / WGSL via wgpu-hal)
pub const VENDOR_APPLE: u32 = 0x106B;

/// ARM Mali GPU
pub const VENDOR_ARM: u32 = 0x13B5;

/// Qualcomm Adreno GPU
pub const VENDOR_QUALCOMM: u32 = 0x5143;

/// ImgTec PowerVR GPU
pub const VENDOR_IMAGINATION: u32 = 0x1010;

/// Software / CPU rasterizer (llvmpipe, SwiftShader, warp)
pub const VENDOR_SOFTWARE: u32 = 0x0000;

/// Return a human-readable vendor name for debug output.
pub fn vendor_name(id: u32) -> &'static str {
    match id {
        VENDOR_NVIDIA => "NVIDIA",
        VENDOR_AMD => "AMD",
        VENDOR_INTEL => "Intel",
        VENDOR_APPLE => "Apple",
        VENDOR_ARM => "ARM Mali",
        VENDOR_QUALCOMM => "Qualcomm Adreno",
        VENDOR_IMAGINATION => "ImgTec PowerVR",
        VENDOR_SOFTWARE => "Software (CPU)",
        _ => "Unknown",
    }
}
