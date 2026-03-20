// SPDX-License-Identifier: AGPL-3.0-or-later
//! Substrate Selection - Explicit Hardware Targeting
//!
//! **Deep Debt Principles**:
//! - ✅ Agnostic & Capability-Based (runtime discovery + selection)
//! - ✅ Modern Idiomatic Rust (enums, pattern matching)
//! - ✅ Safe Rust (zero unsafe)
//! - ✅ Self-Knowledge (substrate discovers own capabilities)
//!
//! **Purpose**: Enable explicit hardware selection for validation and testing

use crate::device::WgpuDevice;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Compute substrate type (hardware target)
///
/// **Deep Debt**: Runtime-discoverable, no hardcoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubstrateType {
    /// CPU execution (any socket)
    Cpu,
    /// NVIDIA GPU (any generation)
    NvidiaGpu,
    /// AMD GPU (any generation)
    AmdGpu,
    /// Intel GPU
    IntelGpu,
    /// Apple GPU (Metal)
    AppleGpu,
    /// NPU (neuromorphic processor)
    Npu,
    /// Other/Unknown
    Other,
}

/// Runtime-discovered compute capability for substrate-level dispatch.
///
/// Inspired by hotSpring metalForge forge `Capability` enum. Code asks
/// "can you do f64?" rather than "are you an RTX 4070?". Distinct from
/// the `unified::Capability` (which covers wgpu feature flags) — this
/// describes what the substrate can *do* at a higher abstraction level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubstrateCapability {
    /// IEEE 754 f64 compute (GPU `SHADER_F64` or CPU native)
    F64Compute,
    /// f32 compute
    F32Compute,
    /// Integer quantized inference at a given bit width (NPU)
    QuantizedInference {
        /// Quantization bit width (e.g. 8 for int8)
        bits: u8,
    },
    /// Batch inference with amortized dispatch (NPU)
    BatchInference {
        /// Maximum batch size for amortized dispatch
        max_batch: u32,
    },
    /// Weight mutation without full reprogramming (NPU)
    WeightMutation,
    /// Scalar reduction pipeline
    ScalarReduce,
    /// Sparse matrix-vector product
    SparseSpMV,
    /// Eigensolve (Lanczos, Jacobi, Householder)
    Eigensolve,
    /// Conjugate gradient solver
    ConjugateGradient,
    /// WGSL shader dispatch via wgpu
    ShaderDispatch,
    /// AVX2/SSE SIMD on CPU
    SimdVector,
    /// GPU timestamp query support
    TimestampQuery,
}

impl SubstrateCapability {
    /// Human-readable label for display.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::F64Compute => "f64",
            Self::F32Compute => "f32",
            Self::QuantizedInference { .. } => "quant",
            Self::BatchInference { .. } => "batch",
            Self::WeightMutation => "weight-mut",
            Self::ScalarReduce => "reduce",
            Self::SparseSpMV => "spmv",
            Self::Eigensolve => "eigen",
            Self::ConjugateGradient => "cg",
            Self::ShaderDispatch => "shader",
            Self::SimdVector => "simd",
            Self::TimestampQuery => "timestamps",
        }
    }
}

/// Specific compute substrate instance
///
/// **Deep Debt**: Capability-based (each substrate knows its capabilities)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Substrate {
    /// Substrate type
    pub substrate_type: SubstrateType,
    /// Human-readable name (e.g., "NVIDIA `GeForce` RTX 3090")
    pub name: String,
    /// Backend (Vulkan, DX12, Metal, OpenGL)
    pub backend: String,
    /// Device index (for multiple instances of same type)
    pub index: usize,
    /// Runtime-discovered capabilities
    pub capabilities: Vec<SubstrateCapability>,
}

impl Substrate {
    /// Create substrate descriptor
    pub fn new(
        substrate_type: SubstrateType,
        name: impl Into<String>,
        backend: impl Into<String>,
        index: usize,
    ) -> Self {
        Self {
            substrate_type,
            name: name.into(),
            backend: backend.into(),
            index,
            capabilities: Vec::new(),
        }
    }

    /// Check if this substrate has a specific capability.
    #[must_use]
    pub fn has(&self, cap: &SubstrateCapability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Summary of capabilities as a comma-separated string.
    pub fn capability_summary(&self) -> String {
        self.capabilities
            .iter()
            .map(SubstrateCapability::label)
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Discover all available substrates (async).
    /// Prefer this over [`discover_all_sync`] when running inside a tokio context
    /// to avoid blocking the executor thread.
    /// # Errors
    /// Returns [`Err`] if wgpu adapter enumeration fails.
    pub async fn discover_all_async() -> Result<Vec<Self>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        Self::build_substrates(adapters)
    }

    /// Discover all available substrates (sync convenience wrapper).
    /// Uses `pollster::block_on` — avoid calling from within an async runtime.
    /// Prefer [`discover_all_async`] in async contexts.
    /// # Errors
    /// Returns [`Err`] if wgpu adapter enumeration fails.
    pub fn discover_all() -> Result<Vec<Self>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        Self::build_substrates(adapters)
    }

    fn build_substrates(adapters: Vec<wgpu::Adapter>) -> Result<Vec<Self>> {
        let mut substrates = Vec::new();
        let mut type_counts = std::collections::HashMap::new();

        for adapter in adapters {
            let info = adapter.get_info();

            // CPU software renderers (llvmpipe, lavapipe) are excluded from
            // substrate discovery — they are development/fallback devices accessed
            // directly via `WgpuDevice`, not production substrates for dispatch.
            if info.device_type == wgpu::DeviceType::Cpu {
                continue;
            }

            // Determine substrate type from vendor and name
            let substrate_type = Self::classify_substrate(&info);

            // Get index for this substrate type
            let index = type_counts.entry(substrate_type).or_insert(0);
            *index += 1;

            let features = adapter.features();
            let mut capabilities = vec![
                SubstrateCapability::F32Compute,
                SubstrateCapability::ShaderDispatch,
            ];
            if features.contains(wgpu::Features::SHADER_F64) {
                capabilities.extend_from_slice(&[
                    SubstrateCapability::F64Compute,
                    SubstrateCapability::ScalarReduce,
                    SubstrateCapability::SparseSpMV,
                    SubstrateCapability::Eigensolve,
                    SubstrateCapability::ConjugateGradient,
                ]);
            }
            if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                capabilities.push(SubstrateCapability::TimestampQuery);
            }

            substrates.push(Self {
                substrate_type,
                name: info.name.clone(),
                backend: format!("{:?}", info.backend),
                index: *index - 1,
                capabilities,
            });
        }

        // Probe for NPU devices (BrainChip AKD1000 via /dev/akida*)
        let akida_path = std::path::Path::new("/dev/akida0");
        if akida_path.exists() {
            let npu_idx = type_counts.entry(SubstrateType::Npu).or_insert(0);
            *npu_idx += 1;
            substrates.push(Self {
                substrate_type: SubstrateType::Npu,
                name: String::from("BrainChip AKD1000"),
                backend: String::from("PCIe"),
                index: *npu_idx - 1,
                capabilities: vec![
                    SubstrateCapability::F32Compute,
                    SubstrateCapability::QuantizedInference { bits: 8 },
                    SubstrateCapability::QuantizedInference { bits: 4 },
                    SubstrateCapability::BatchInference { max_batch: 8 },
                    SubstrateCapability::WeightMutation,
                ],
            });
        }

        Ok(substrates)
    }

    /// Classify substrate type from adapter info using PCI vendor IDs.
    /// Primary detection uses standard PCI vendor IDs (reliable across drivers).
    /// Falls back to name-based detection for non-PCI devices (Apple, NPU).
    fn classify_substrate(info: &wgpu::AdapterInfo) -> SubstrateType {
        use super::vendor::{VENDOR_AMD, VENDOR_APPLE, VENDOR_INTEL, VENDOR_NVIDIA};

        match info.vendor {
            VENDOR_NVIDIA => SubstrateType::NvidiaGpu,
            VENDOR_AMD => SubstrateType::AmdGpu,
            VENDOR_INTEL => SubstrateType::IntelGpu,
            VENDOR_APPLE => SubstrateType::AppleGpu,
            _ => {
                let name_lower = info.name.to_lowercase();
                if name_lower.contains("apple")
                    || name_lower.contains("m1")
                    || name_lower.contains("m2")
                    || name_lower.contains("m3")
                {
                    SubstrateType::AppleGpu
                } else if name_lower.contains("npu") || name_lower.contains("akida") {
                    SubstrateType::Npu
                } else {
                    SubstrateType::Other
                }
            }
        }
    }

    /// Create `WgpuDevice` on this specific substrate
    /// **Deep Debt**: Explicit selection, no implicit behavior
    /// Multi-device support: When multiple devices of the same type exist,
    /// selects the device at the specified index within that type.
    /// # Errors
    /// Returns [`Err`] if no adapter matches the substrate type/index, or if
    /// device creation fails (e.g., driver initialization, required features not supported).
    pub async fn create_device(&self) -> Result<WgpuDevice> {
        let substrate_type = self.substrate_type;
        let target_index = self.index;

        // Counter to track which device of this type we're on
        let device_counter = AtomicUsize::new(0);

        WgpuDevice::new_with_filter(wgpu::Backends::all(), move |info| {
            let detected_type = Self::classify_substrate(info);
            if detected_type != substrate_type {
                return false;
            }

            // Multi-device index matching: select Nth device of this type
            let current_index = device_counter.fetch_add(1, Ordering::SeqCst);
            current_index == target_index
        })
        .await
    }
}

impl std::fmt::Display for Substrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}[{}]: {} ({})",
            self.substrate_type, self.index, self.name, self.backend
        )
    }
}

impl std::fmt::Display for SubstrateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstrateType::Cpu => write!(f, "CPU"),
            SubstrateType::NvidiaGpu => write!(f, "NVIDIA GPU"),
            SubstrateType::AmdGpu => write!(f, "AMD GPU"),
            SubstrateType::IntelGpu => write!(f, "Intel GPU"),
            SubstrateType::AppleGpu => write!(f, "Apple GPU"),
            SubstrateType::Npu => write!(f, "NPU"),
            SubstrateType::Other => write!(f, "Other"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_substrate_discovery() {
        let substrates = Substrate::discover_all().unwrap();
        println!("Discovered {} substrates:", substrates.len());
        for substrate in &substrates {
            println!("  - {substrate}");
        }
        assert!(
            !substrates.is_empty(),
            "Should discover at least one substrate"
        );
    }

    #[tokio::test]
    async fn test_substrate_device_creation() {
        let substrates = Substrate::discover_all().unwrap();
        if let Some(substrate) = substrates.first() {
            println!("Testing device creation on: {substrate}");
            let device = substrate.create_device().await.unwrap();
            println!("✓ Created device: {}", device.name());
        }
    }
}
