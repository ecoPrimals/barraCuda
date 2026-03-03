// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compute Kernel Routing - Unified Math Language → Hardware-Specific Implementation
//!
//! **Evolution (Feb 15, 2026)**: Completes the NPU pipeline!
//!
//! This module implements the "compile/adjust" layer for routing math
//! to the appropriate hardware:
//!
//! - **GPU/CPU**: WGSL shaders run directly (our unified math language)
//! - **NPU**: Route to pre-compiled models if workload matches NPU capability
//! - **TPU**: (Future) Route to TPU-specific operations
//!
//! # Philosophy
//!
//! barraCuda treats WGSL as the "unified math language" for arbitrary computation.
//! When a workload can be accelerated by specialized hardware, we route to
//! hardware-specific implementations. Otherwise, WGSL runs on GPU/CPU.
//!
//! This is NOT a compiler (WGSL → NPU bytecode). NPUs run pre-compiled models,
//! not arbitrary code. The routing layer matches workloads to available
//! hardware capabilities.
//!
//! # Example
//!
//! ```no_run
//! use barracuda::device::kernel_router::{KernelRouter, ComputeWorkload, KernelTarget};
//!
//! # fn main() -> barracuda::error::Result<()> {
//! let router = KernelRouter::new()?;
//! let workload = ComputeWorkload::SparseInference {
//!     input_sparsity: 0.95,
//!     model_name: "edge_classifier".to_string(),
//! };
//! let target = router.route(&workload)?;
//! match target {
//!     KernelTarget::Npu { model_id: _, .. } => {}
//!     KernelTarget::Wgsl { shader: _, device: _, .. } => {}
//!     KernelTarget::Hybrid { .. } => {}
//! }
//! # Ok(())
//! # }
//! ```

use crate::device::{Device, DeviceSelection};
use crate::error::Result;
use std::collections::HashMap;

/// Workloads smaller than this use CPU to avoid GPU dispatch overhead.
const CPU_FALLBACK_THRESHOLD: usize = 1_000;

/// Eigendecomposition matrices smaller than this route to CPU.
const EIGENDECOMP_CPU_THRESHOLD: usize = 128;

/// Linear systems smaller than this route to CPU.
const LINEAR_SOLVE_CPU_THRESHOLD: usize = 256;

/// Matmul dimensions above which 16x16 tile workgroup is used.
const MATMUL_LARGE_DIM: usize = 256;

/// Matmul dimensions above which 8x8 tile workgroup is used.
const MATMUL_MEDIUM_DIM: usize = 64;

/// Compute workload description
///
/// Describes WHAT computation needs to be done, not HOW.
/// The router decides which hardware to target.
#[derive(Debug, Clone)]
pub enum ComputeWorkload {
    /// Dense matrix operations (matmul, conv) - always GPU/CPU
    DenseMatmul {
        /// Matrix dimensions for tuning
        m: usize,
        n: usize,
        k: usize,
    },

    /// Sparse inference with high input sparsity - NPU preferred
    SparseInference {
        /// Input sparsity ratio (0.0 = dense, 1.0 = empty)
        input_sparsity: f64,
        /// Model identifier (for NPU model lookup)
        model_name: String,
    },

    /// Reservoir computing / Echo State Network - NPU natural fit
    ReservoirState {
        /// Reservoir size
        reservoir_size: usize,
        /// Input dimension
        input_dim: usize,
    },

    /// FFT computation - always GPU (butterfly stages are parallel)
    FFT { size: usize, batch_count: usize },

    /// Physics force computation - always GPU (needs arbitrary WGSL)
    PhysicsForce {
        particle_count: usize,
        force_type: String,
    },

    /// Eigenvalue decomposition - GPU for large, CPU for small
    Eigendecomp { matrix_size: usize },

    /// Linear system solve - GPU for large, CPU for small
    LinearSolve { system_size: usize },

    /// Binary pre-screening / filtering - NPU ideal (ultra-low power)
    BinaryPrescreen { input_count: usize, threshold: f64 },

    /// Generic WGSL operation - always GPU/CPU
    GenericWgsl { shader_name: String },
}

/// Target hardware for kernel execution
#[derive(Debug, Clone)]
pub enum KernelTarget {
    /// Run on NPU with pre-compiled model
    Npu {
        /// Model file path or ID
        model_id: String,
        /// Expected input shape
        input_shape: Vec<usize>,
        /// Expected output shape
        output_shape: Vec<usize>,
    },

    /// Run WGSL shader on GPU or CPU
    Wgsl {
        /// Shader name/path
        shader: String,
        /// Target device (GPU or CPU)
        device: DeviceSelection,
        /// Workgroup size hint
        workgroup_size: [u32; 3],
    },

    /// Hybrid: NPU for preprocessing, GPU for main compute
    Hybrid {
        /// NPU model for sparse filtering
        npu_prefilter: Option<String>,
        /// WGSL shader for dense compute
        wgsl_compute: String,
        /// Target device for WGSL
        device: DeviceSelection,
    },
}

/// Kernel router - maps workloads to hardware
///
/// The router knows what hardware is available and which workloads
/// each device can accelerate. It returns the best target for a given
/// computation.
pub struct KernelRouter {
    /// Available NPU models (model_name → model_path)
    npu_models: HashMap<String, NpuModelInfo>,
    /// GPU available
    has_gpu: bool,
    /// NPU available
    has_npu: bool,
    has_tpu: bool,
}

/// NPU model metadata
#[derive(Debug, Clone)]
pub struct NpuModelInfo {
    /// Model file path
    pub path: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Sparsity threshold for efficient execution
    pub sparsity_threshold: f64,
}

impl KernelRouter {
    /// Create new kernel router with runtime hardware discovery
    pub fn new() -> Result<Self> {
        let has_gpu = Device::GPU.is_available();
        let has_npu = Device::NPU.is_available();
        let has_tpu = Device::TPU.is_available();

        // Discover available NPU models
        let npu_models = Self::discover_npu_models()?;

        Ok(Self {
            npu_models,
            has_gpu,
            has_npu,
            has_tpu,
        })
    }

    /// Whether a TPU is available for routing.
    pub fn has_tpu(&self) -> bool {
        self.has_tpu
    }

    /// Route workload to best hardware target
    pub fn route(&self, workload: &ComputeWorkload) -> Result<KernelTarget> {
        match workload {
            // === Dense compute → Always GPU/CPU (WGSL) ===
            ComputeWorkload::DenseMatmul { m, n, k } => {
                let size = m * n * k;
                let device = self.select_wgsl_device(size);
                Ok(KernelTarget::Wgsl {
                    shader: "matmul".to_string(),
                    device,
                    workgroup_size: self.optimal_workgroup_for_matmul(*m, *n, *k),
                })
            }

            ComputeWorkload::FFT { size, batch_count } => {
                let total = size * batch_count;
                let device = self.select_wgsl_device(total);
                Ok(KernelTarget::Wgsl {
                    shader: "fft".to_string(),
                    device,
                    workgroup_size: [256, 1, 1],
                })
            }

            ComputeWorkload::PhysicsForce {
                particle_count,
                force_type,
            } => {
                let device = self.select_wgsl_device(*particle_count);
                Ok(KernelTarget::Wgsl {
                    shader: format!("forces/{force_type}"),
                    device,
                    workgroup_size: [64, 1, 1],
                })
            }

            ComputeWorkload::Eigendecomp { matrix_size } => {
                let device = if *matrix_size < EIGENDECOMP_CPU_THRESHOLD {
                    DeviceSelection::Cpu
                } else {
                    self.select_wgsl_device(matrix_size * matrix_size)
                };
                Ok(KernelTarget::Wgsl {
                    shader: "linalg/eigh".to_string(),
                    device,
                    workgroup_size: [16, 16, 1],
                })
            }

            ComputeWorkload::LinearSolve { system_size } => {
                let device = if *system_size < LINEAR_SOLVE_CPU_THRESHOLD {
                    DeviceSelection::Cpu
                } else {
                    self.select_wgsl_device(system_size * system_size)
                };
                Ok(KernelTarget::Wgsl {
                    shader: "linalg/solve".to_string(),
                    device,
                    workgroup_size: [16, 16, 1],
                })
            }

            ComputeWorkload::GenericWgsl { shader_name } => {
                let device = self.select_wgsl_device(CPU_FALLBACK_THRESHOLD);
                Ok(KernelTarget::Wgsl {
                    shader: shader_name.clone(),
                    device,
                    workgroup_size: [64, 1, 1],
                })
            }

            // === Sparse/Event workloads → NPU preferred if available ===
            ComputeWorkload::SparseInference {
                input_sparsity,
                model_name,
            } => {
                // NPU: if available AND we have the model AND sparsity is high enough
                if self.has_npu {
                    if let Some(model_info) = self.npu_models.get(model_name) {
                        if *input_sparsity >= model_info.sparsity_threshold {
                            return Ok(KernelTarget::Npu {
                                model_id: model_info.path.clone(),
                                input_shape: model_info.input_shape.clone(),
                                output_shape: model_info.output_shape.clone(),
                            });
                        }
                    }
                }
                let device = self.select_wgsl_device(CPU_FALLBACK_THRESHOLD * 10);
                Ok(KernelTarget::Wgsl {
                    shader: format!("snn/{model_name}"),
                    device,
                    workgroup_size: [64, 1, 1],
                })
            }

            ComputeWorkload::ReservoirState {
                reservoir_size,
                input_dim,
            } => {
                // NPU: natural fit for reservoir computing (event-driven)
                if self.has_npu {
                    if let Some(model_info) = self.npu_models.get("reservoir") {
                        return Ok(KernelTarget::Npu {
                            model_id: model_info.path.clone(),
                            input_shape: vec![*input_dim],
                            output_shape: vec![*reservoir_size],
                        });
                    }
                }
                // Fallback: WGSL reservoir simulation
                let device = self.select_wgsl_device(reservoir_size * input_dim);
                Ok(KernelTarget::Wgsl {
                    shader: "reservoir/esn".to_string(),
                    device,
                    workgroup_size: [64, 1, 1],
                })
            }

            ComputeWorkload::BinaryPrescreen {
                input_count,
                threshold: _,
            } => {
                // NPU: ideal for binary pre-screening (ultra-low power)
                if self.has_npu {
                    if let Some(model_info) = self.npu_models.get("prescreen") {
                        return Ok(KernelTarget::Npu {
                            model_id: model_info.path.clone(),
                            input_shape: vec![*input_count],
                            output_shape: vec![*input_count],
                        });
                    }
                }
                // Fallback: simple threshold on GPU
                let device = self.select_wgsl_device(*input_count);
                Ok(KernelTarget::Wgsl {
                    shader: "prescreen/binary_threshold".to_string(),
                    device,
                    workgroup_size: [256, 1, 1],
                })
            }
        }
    }

    /// Check if a workload CAN be accelerated by NPU
    pub fn can_route_to_npu(&self, workload: &ComputeWorkload) -> bool {
        if !self.has_npu {
            return false;
        }

        match workload {
            ComputeWorkload::SparseInference { model_name, .. } => {
                self.npu_models.contains_key(model_name)
            }
            ComputeWorkload::ReservoirState { .. } => self.npu_models.contains_key("reservoir"),
            ComputeWorkload::BinaryPrescreen { .. } => self.npu_models.contains_key("prescreen"),
            // Dense compute cannot go to NPU
            _ => false,
        }
    }

    /// Select best WGSL device (GPU or CPU)
    fn select_wgsl_device(&self, workload_size: usize) -> DeviceSelection {
        if workload_size < CPU_FALLBACK_THRESHOLD {
            return DeviceSelection::Cpu;
        }
        if self.has_gpu {
            DeviceSelection::Gpu
        } else {
            DeviceSelection::Cpu
        }
    }

    /// Optimal workgroup size for matmul
    fn optimal_workgroup_for_matmul(&self, m: usize, n: usize, _k: usize) -> [u32; 3] {
        if m >= MATMUL_LARGE_DIM && n >= MATMUL_LARGE_DIM {
            [16, 16, 1]
        } else if m >= MATMUL_MEDIUM_DIM && n >= MATMUL_MEDIUM_DIM {
            [8, 8, 1]
        } else {
            [4, 4, 1]
        }
    }

    /// Discover available NPU models from filesystem
    fn discover_npu_models() -> Result<HashMap<String, NpuModelInfo>> {
        let mut models = HashMap::new();

        // Check standard model locations (absolute paths only)
        let mut model_dirs = vec![
            "/usr/share/akida/models".to_string(),
            "/opt/akida/models".to_string(),
        ];

        // Add home directory path if available
        if let Ok(home) = std::env::var("HOME") {
            model_dirs.push(format!("{home}/.local/share/akida/models"));
        }

        for dir in model_dirs {
            let path = std::path::Path::new(&dir);
            if path.exists() {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        if let Some(ext) = entry.path().extension() {
                            if ext == "fbz" {
                                // Found a model - extract metadata
                                if let Some(name) = entry.path().file_stem() {
                                    let model_name = name.to_string_lossy().to_string();
                                    models.insert(
                                        model_name.clone(),
                                        NpuModelInfo {
                                            path: entry.path().to_string_lossy().to_string(),
                                            // Shapes would come from model metadata in production
                                            input_shape: vec![128],
                                            output_shape: vec![10],
                                            sparsity_threshold: 0.7,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(models)
    }

    /// Register a custom NPU model
    pub fn register_npu_model(&mut self, name: &str, info: NpuModelInfo) {
        self.npu_models.insert(name.to_string(), info);
    }

    /// Get list of available NPU models
    pub fn available_npu_models(&self) -> Vec<&str> {
        self.npu_models.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for KernelRouter {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            npu_models: HashMap::new(),
            has_gpu: true,
            has_npu: false,
            has_tpu: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_router_creation() {
        let router = KernelRouter::new();
        assert!(router.is_ok());
    }

    #[test]
    fn test_dense_matmul_routes_to_wgsl() {
        let router = KernelRouter::default();
        let workload = ComputeWorkload::DenseMatmul {
            m: 1024,
            n: 1024,
            k: 1024,
        };
        let target = router.route(&workload).unwrap();
        match target {
            KernelTarget::Wgsl { shader, device, .. } => {
                assert_eq!(shader, "matmul");
                assert!(device.supports_wgsl());
            }
            _ => panic!("Dense matmul should route to WGSL"),
        }
    }

    #[test]
    fn test_physics_force_routes_to_wgsl() {
        let router = KernelRouter::default();
        let workload = ComputeWorkload::PhysicsForce {
            particle_count: 10_000,
            force_type: "lennard_jones".to_string(),
        };
        let target = router.route(&workload).unwrap();
        match target {
            KernelTarget::Wgsl { device, .. } => {
                assert!(device.supports_wgsl());
            }
            _ => panic!("Physics force should route to WGSL"),
        }
    }

    #[test]
    fn test_small_eigendecomp_routes_to_cpu() {
        let router = KernelRouter::default();
        let workload = ComputeWorkload::Eigendecomp { matrix_size: 32 };
        let target = router.route(&workload).unwrap();
        match target {
            KernelTarget::Wgsl { device, .. } => {
                assert_eq!(device, DeviceSelection::Cpu);
            }
            _ => panic!("Small eigendecomp should route to CPU"),
        }
    }

    #[test]
    fn test_sparse_inference_fallback() {
        let router = KernelRouter::default(); // No NPU models registered
        let workload = ComputeWorkload::SparseInference {
            input_sparsity: 0.95,
            model_name: "nonexistent".to_string(),
        };
        let target = router.route(&workload).unwrap();
        // Should fall back to WGSL
        match target {
            KernelTarget::Wgsl { .. } => {}
            _ => panic!("Missing NPU model should fall back to WGSL"),
        }
    }

    #[test]
    fn test_can_route_to_npu() {
        let router = KernelRouter::default();

        // Without NPU models, nothing can route to NPU
        let workload = ComputeWorkload::SparseInference {
            input_sparsity: 0.95,
            model_name: "test".to_string(),
        };
        // NPU not available in default router
        assert!(!router.can_route_to_npu(&workload));

        // Dense matmul can never route to NPU
        let dense = ComputeWorkload::DenseMatmul {
            m: 1024,
            n: 1024,
            k: 1024,
        };
        assert!(!router.can_route_to_npu(&dense));
    }

    #[test]
    fn test_register_npu_model() {
        let mut router = KernelRouter::default();
        router.register_npu_model(
            "custom_model",
            NpuModelInfo {
                path: "/path/to/model.fbz".to_string(),
                input_shape: vec![128],
                output_shape: vec![10],
                sparsity_threshold: 0.8,
            },
        );
        assert!(router.available_npu_models().contains(&"custom_model"));
    }
}
