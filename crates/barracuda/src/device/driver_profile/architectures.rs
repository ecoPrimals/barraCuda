// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU microarchitecture detection — name-based identification of GPU families.
//!
//! Maps adapter names (e.g. "NVIDIA `GeForce` RTX 4090") to `GpuArch` for
//! latency models, FP64 rate heuristics, and workgroup sizing.

use crate::device::WgpuDevice;

/// GPU microarchitecture generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArch {
    /// NVIDIA Volta (SM70) — Titan V, Quadro GV100
    Volta,
    /// NVIDIA Turing (SM75) — RTX 2000 series
    Turing,
    /// NVIDIA Ampere (SM80/86) — RTX 3000 series
    Ampere,
    /// NVIDIA Ada Lovelace (SM89) — RTX 4000 series
    Ada,
    /// NVIDIA Blackwell (SM100/SM120) — RTX 5000 series
    Blackwell,
    /// AMD RDNA 2 — RX 6000 series
    Rdna2,
    /// AMD RDNA 3 — RX 7000 series
    Rdna3,
    /// AMD CDNA 2 — MI200 series
    Cdna2,
    /// Intel Arc (Alchemist/Battlemage)
    IntelArc,
    /// Apple M-series GPU (Apple Silicon — M1/M2/M3/M4 family)
    ///
    /// Runs via Metal + wgpu's Metal backend. FP64 is emulated in software
    /// (Apple GPUs only have f32 hardware). ILP window empirically ~4 cycles.
    AppleM,
    /// Software rasterizer
    Software,
    /// Unknown GPU architecture
    Unknown,
}

/// Detect GPU architecture from adapter name string.
///
/// Used by [`detect_arch`]; exposed for unit testing.
#[must_use]
pub(crate) fn arch_from_adapter_name(name: &str) -> GpuArch {
    let name = name.to_lowercase();

    if name.contains("titan v") || name.contains("gv100") || name.contains("v100") {
        return GpuArch::Volta;
    }
    if name.contains("rtx 20") || name.contains("rtx20") || name.contains("tu1") {
        return GpuArch::Turing;
    }
    if name.contains("rtx 30") || name.contains("rtx30") || name.contains("a100") {
        return GpuArch::Ampere;
    }
    if name.contains("rtx 40") || name.contains("rtx40") || name.contains("l40") {
        return GpuArch::Ada;
    }
    if name.contains("rtx 50")
        || name.contains("rtx50")
        || name.contains("gb2")
        || name.contains("b200")
        || name.contains("b100")
    {
        return GpuArch::Blackwell;
    }

    if name.contains("rx 6") || name.contains("rx6") {
        return GpuArch::Rdna2;
    }
    if name.contains("rx 7") || name.contains("rx7") {
        return GpuArch::Rdna3;
    }
    if name.contains("mi2") || name.contains("mi3") {
        return GpuArch::Cdna2;
    }

    if name.contains("arc") || name.contains("a770") || name.contains("a750") {
        return GpuArch::IntelArc;
    }

    if name.contains("apple m") || name.contains("apple paravirtual") {
        return GpuArch::AppleM;
    }

    if name.contains("llvmpipe") || name.contains("swiftshader") {
        return GpuArch::Software;
    }

    GpuArch::Unknown
}

/// Detect GPU architecture from device adapter name.
pub(crate) fn detect_arch(device: &WgpuDevice) -> GpuArch {
    arch_from_adapter_name(&device.adapter_info().name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackwell_detection_rtx_50() {
        assert_eq!(
            arch_from_adapter_name("NVIDIA GeForce RTX 5060"),
            GpuArch::Blackwell
        );
        assert_eq!(
            arch_from_adapter_name("NVIDIA GeForce RTX 50"),
            GpuArch::Blackwell
        );
    }

    #[test]
    fn blackwell_detection_rtx50_no_space() {
        assert_eq!(
            arch_from_adapter_name("NVIDIA RTX50 Series"),
            GpuArch::Blackwell
        );
    }

    #[test]
    fn blackwell_detection_gb2() {
        assert_eq!(arch_from_adapter_name("NVIDIA GB200"), GpuArch::Blackwell);
        assert_eq!(arch_from_adapter_name("NVIDIA GB2"), GpuArch::Blackwell);
    }

    #[test]
    fn blackwell_detection_b200_b100() {
        assert_eq!(arch_from_adapter_name("NVIDIA B200"), GpuArch::Blackwell);
        assert_eq!(arch_from_adapter_name("NVIDIA B100"), GpuArch::Blackwell);
    }
}
