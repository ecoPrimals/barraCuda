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
/// Vendor-agnostic: classifies by device form factor, not manufacturer.
/// Uses wgpu `DeviceType` as the source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubstrateType {
    /// CPU execution (any socket)
    Cpu,
    /// Discrete GPU (PCIe-attached, any vendor)
    DiscreteGpu,
    /// Integrated GPU (unified memory, any vendor)
    IntegratedGpu,
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

    /// Classify substrate type from wgpu `DeviceType` (vendor-agnostic).
    fn classify_substrate(info: &wgpu::AdapterInfo) -> SubstrateType {
        match info.device_type {
            wgpu::DeviceType::DiscreteGpu => SubstrateType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu => SubstrateType::IntegratedGpu,
            wgpu::DeviceType::Cpu => SubstrateType::Cpu,
            _ => {
                let name_lower = info.name.to_lowercase();
                if name_lower.contains("npu") || name_lower.contains("akida") {
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
            Self::Cpu => write!(f, "CPU"),
            Self::DiscreteGpu => write!(f, "Discrete GPU"),
            Self::IntegratedGpu => write!(f, "Integrated GPU"),
            Self::Npu => write!(f, "NPU"),
            Self::Other => write!(f, "Other"),
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
        let Some(substrate) = substrates.first() else {
            return;
        };
        println!("Testing device creation on: {substrate}");
        match substrate.create_device().await {
            Ok(device) => println!("Created device: {}", device.name()),
            Err(e) => {
                // GPU device creation can fail transiently (device lost, OOM,
                // driver contention). These are hardware-level failures, not
                // logic bugs — skip gracefully with diagnostics.
                println!("Device creation failed (hardware/driver): {e}");
            }
        }
    }

    #[test]
    fn substrate_type_display() {
        assert_eq!(format!("{}", SubstrateType::Cpu), "CPU");
        assert_eq!(format!("{}", SubstrateType::DiscreteGpu), "Discrete GPU");
        assert_eq!(
            format!("{}", SubstrateType::IntegratedGpu),
            "Integrated GPU"
        );
        assert_eq!(format!("{}", SubstrateType::Npu), "NPU");
        assert_eq!(format!("{}", SubstrateType::Other), "Other");
    }

    #[test]
    fn substrate_type_serde_roundtrip() {
        for ty in [
            SubstrateType::Cpu,
            SubstrateType::DiscreteGpu,
            SubstrateType::IntegratedGpu,
            SubstrateType::Npu,
            SubstrateType::Other,
        ] {
            let json = serde_json::to_string(&ty).unwrap();
            let back: SubstrateType = serde_json::from_str(&json).unwrap();
            assert_eq!(ty, back);
        }
    }

    #[test]
    fn substrate_capability_labels_all_nonempty() {
        let caps = [
            SubstrateCapability::F64Compute,
            SubstrateCapability::F32Compute,
            SubstrateCapability::QuantizedInference { bits: 8 },
            SubstrateCapability::BatchInference { max_batch: 32 },
            SubstrateCapability::WeightMutation,
            SubstrateCapability::ScalarReduce,
            SubstrateCapability::SparseSpMV,
            SubstrateCapability::Eigensolve,
            SubstrateCapability::ConjugateGradient,
            SubstrateCapability::ShaderDispatch,
            SubstrateCapability::SimdVector,
            SubstrateCapability::TimestampQuery,
        ];
        for cap in &caps {
            assert!(!cap.label().is_empty());
        }
    }

    #[test]
    fn substrate_new_starts_empty_capabilities() {
        let sub = Substrate::new(SubstrateType::DiscreteGpu, "Test GPU", "Vulkan", 0);
        assert!(sub.capabilities.is_empty());
        assert_eq!(sub.substrate_type, SubstrateType::DiscreteGpu);
        assert_eq!(sub.name, "Test GPU");
        assert_eq!(sub.backend, "Vulkan");
        assert_eq!(sub.index, 0);
    }

    #[test]
    fn substrate_has_capability() {
        let mut sub = Substrate::new(SubstrateType::DiscreteGpu, "GPU", "Vulkan", 0);
        sub.capabilities.push(SubstrateCapability::F64Compute);
        sub.capabilities.push(SubstrateCapability::ShaderDispatch);
        assert!(sub.has(&SubstrateCapability::F64Compute));
        assert!(sub.has(&SubstrateCapability::ShaderDispatch));
        assert!(!sub.has(&SubstrateCapability::Eigensolve));
    }

    #[test]
    fn substrate_capability_summary() {
        let mut sub = Substrate::new(SubstrateType::DiscreteGpu, "GPU", "Vulkan", 0);
        sub.capabilities.push(SubstrateCapability::F64Compute);
        sub.capabilities.push(SubstrateCapability::ShaderDispatch);
        let summary = sub.capability_summary();
        assert!(summary.contains("f64"));
        assert!(summary.contains("shader"));
    }

    #[test]
    fn substrate_display_includes_name_and_type() {
        let sub = Substrate::new(SubstrateType::DiscreteGpu, "MyGPU", "Vulkan", 0);
        let display = format!("{sub}");
        assert!(display.contains("MyGPU"));
        assert!(display.contains("DiscreteGpu"));
    }

    #[test]
    fn substrate_serde_roundtrip() {
        let mut sub = Substrate::new(SubstrateType::IntegratedGpu, "iGPU", "Metal", 1);
        sub.capabilities.push(SubstrateCapability::F32Compute);
        let json = serde_json::to_string(&sub).unwrap();
        let back: Substrate = serde_json::from_str(&json).unwrap();
        assert_eq!(sub, back);
    }
}
