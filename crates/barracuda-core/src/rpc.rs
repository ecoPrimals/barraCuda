// SPDX-License-Identifier: AGPL-3.0-or-later
//! tarpc service definition for barraCuda.
//!
//! High-performance binary RPC alongside the JSON-RPC 2.0 text protocol.
//! Per wateringHole `UNIVERSAL_IPC_STANDARD_V3.md`, tarpc is the optional
//! high-throughput protocol for primal-to-primal calls. JSON-RPC 2.0 remains
//! the primary protocol for external/cross-language consumers.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda_core::rpc::BarraCudaClient;
//!
//! let client = BarraCudaClient::new(/* ... */);
//! let devices = client.device_list(context::current()).await?;
//! ```

use serde::{Deserialize, Serialize};

/// Device information returned by `device_list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Adapter name (e.g. "NVIDIA GeForce RTX 3090").
    pub name: String,
    /// Vendor identifier.
    pub vendor: u32,
    /// Device type (Discrete, Integrated, Cpu, etc.).
    pub device_type: String,
    /// Backend (Vulkan, Metal, DX12, Gl).
    pub backend: String,
    /// Driver name.
    pub driver: String,
    /// Driver version info.
    pub driver_info: String,
}

/// Device probe result from `device_probe`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProbe {
    /// Whether a device is available.
    pub available: bool,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: u64,
    /// Max storage buffers per shader stage.
    pub max_storage_buffers: u32,
    /// Max compute workgroup size X.
    pub max_workgroup_size_x: u32,
    /// Max compute workgroups per dimension.
    pub max_workgroups_per_dimension: u32,
}

/// Health report from `health_check`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Primal name.
    pub name: String,
    /// Primal version.
    pub version: String,
    /// Status string.
    pub status: String,
}

/// Tolerance pair for numerical operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tolerances {
    /// Tolerance name.
    pub name: String,
    /// Absolute tolerance.
    pub abs_tol: f64,
    /// Relative tolerance.
    pub rel_tol: f64,
}

/// Tensor creation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorHandle {
    /// Unique tensor identifier.
    pub tensor_id: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Number of elements.
    pub elements: usize,
    /// Data type.
    pub dtype: String,
}

/// GPU validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether GPU is available.
    pub gpu_available: bool,
    /// Validation status.
    pub status: String,
    /// Human-readable message.
    pub message: String,
}

/// Compute dispatch result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchResult {
    /// Status of the dispatch.
    pub status: String,
    /// Shader that was dispatched.
    pub shader: String,
    /// Entry point used.
    pub entry_point: String,
}

/// FHE operation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FheResult {
    /// Status of the operation.
    pub status: String,
    /// Modulus used.
    pub modulus: u64,
}

/// Matmul result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulResult {
    /// Status of the operation.
    pub status: String,
    /// Left-hand tensor ID.
    pub lhs_id: String,
    /// Right-hand tensor ID.
    pub rhs_id: String,
}

/// barraCuda tarpc service.
///
/// Mirrors the JSON-RPC 2.0 endpoints with strongly-typed signatures.
/// All methods follow the semantic naming standard: `barracuda.{domain}.{operation}`.
#[tarpc::service]
pub trait BarraCudaService {
    /// List available compute devices.
    async fn device_list() -> Vec<DeviceInfo>;

    /// Probe device capabilities.
    async fn device_probe() -> DeviceProbe;

    /// Health check.
    async fn health_check() -> HealthReport;

    /// Get numerical tolerances for a named operation.
    async fn tolerances_get(name: String) -> Tolerances;

    /// Run GPU stack validation.
    async fn validate_gpu_stack() -> ValidationResult;

    /// Dispatch a compute shader.
    async fn compute_dispatch(shader: String, entry_point: String) -> DispatchResult;

    /// Create a tensor on the device.
    async fn tensor_create(shape: Vec<usize>, dtype: String) -> TensorHandle;

    /// Matrix multiply two tensors.
    async fn tensor_matmul(lhs_id: String, rhs_id: String) -> MatmulResult;

    /// FHE Number Theoretic Transform.
    async fn fhe_ntt(modulus: u64, degree: u64) -> FheResult;

    /// FHE pointwise polynomial multiplication.
    async fn fhe_pointwise_mul(modulus: u64) -> FheResult;
}

/// tarpc server implementation that delegates to `BarraCudaPrimal`.
#[derive(Clone)]
pub struct BarraCudaServer {
    primal: std::sync::Arc<crate::BarraCudaPrimal>,
}

impl BarraCudaServer {
    /// Create a new tarpc server wrapping the primal.
    pub fn new(primal: std::sync::Arc<crate::BarraCudaPrimal>) -> Self {
        Self { primal }
    }
}

impl BarraCudaService for BarraCudaServer {
    async fn device_list(self, _: tarpc::context::Context) -> Vec<DeviceInfo> {
        let mut devices = Vec::new();
        if let Some(dev) = self.primal.device() {
            let info = dev.adapter_info();
            devices.push(DeviceInfo {
                name: info.name.clone(),
                vendor: info.vendor,
                device_type: format!("{:?}", info.device_type),
                backend: format!("{:?}", info.backend),
                driver: info.driver.clone(),
                driver_info: info.driver_info.clone(),
            });
        }
        devices
    }

    async fn device_probe(self, _: tarpc::context::Context) -> DeviceProbe {
        match self.primal.device() {
            Some(dev) => {
                let limits = dev.device().limits();
                DeviceProbe {
                    available: true,
                    max_buffer_size: limits.max_buffer_size,
                    max_storage_buffers: limits.max_storage_buffers_per_shader_stage,
                    max_workgroup_size_x: limits.max_compute_workgroup_size_x,
                    max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                }
            }
            None => DeviceProbe {
                available: false,
                max_buffer_size: 0,
                max_storage_buffers: 0,
                max_workgroup_size_x: 0,
                max_workgroups_per_dimension: 0,
            },
        }
    }

    async fn health_check(self, _: tarpc::context::Context) -> HealthReport {
        use sourdough_core::PrimalHealth;
        match self.primal.health_check().await {
            Ok(report) => HealthReport {
                name: report.name.clone(),
                version: report.version.clone(),
                status: format!("{:?}", report.status),
            },
            Err(e) => HealthReport {
                name: "barraCuda".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                status: format!("Error: {e}"),
            },
        }
    }

    async fn tolerances_get(self, _: tarpc::context::Context, name: String) -> Tolerances {
        let (abs_tol, rel_tol) = match name.as_str() {
            "fhe" => (0.0, 0.0),
            "f64" | "double" => (1e-12, 1e-10),
            "f32" | "float" => (1e-5, 1e-4),
            "df64" | "emulated_double" => (1e-10, 1e-8),
            _ => (1e-6, 1e-5),
        };
        Tolerances {
            name,
            abs_tol,
            rel_tol,
        }
    }

    async fn validate_gpu_stack(self, _: tarpc::context::Context) -> ValidationResult {
        ValidationResult {
            gpu_available: self.primal.device().is_some(),
            status: if self.primal.device().is_some() {
                "validation_available".to_string()
            } else {
                "no_device".to_string()
            },
            message: "Use `barracuda validate` CLI for full GPU stack validation".to_string(),
        }
    }

    async fn compute_dispatch(
        self,
        _: tarpc::context::Context,
        shader: String,
        entry_point: String,
    ) -> DispatchResult {
        DispatchResult {
            status: if self.primal.device().is_some() {
                "accepted"
            } else {
                "no_device"
            }
            .to_string(),
            shader,
            entry_point,
        }
    }

    async fn tensor_create(
        self,
        _: tarpc::context::Context,
        shape: Vec<usize>,
        dtype: String,
    ) -> TensorHandle {
        let elements: usize = shape.iter().product();
        let tensor_id = blake3::hash(format!("tensor:{shape:?}:{elements}").as_bytes()).to_hex()
            [..16]
            .to_string();

        TensorHandle {
            tensor_id,
            shape,
            elements,
            dtype,
        }
    }

    async fn tensor_matmul(
        self,
        _: tarpc::context::Context,
        lhs_id: String,
        rhs_id: String,
    ) -> MatmulResult {
        MatmulResult {
            status: if self.primal.device().is_some() {
                "accepted"
            } else {
                "no_device"
            }
            .to_string(),
            lhs_id,
            rhs_id,
        }
    }

    async fn fhe_ntt(self, _: tarpc::context::Context, modulus: u64, _degree: u64) -> FheResult {
        FheResult {
            status: if self.primal.device().is_some() {
                "accepted"
            } else {
                "no_device"
            }
            .to_string(),
            modulus,
        }
    }

    async fn fhe_pointwise_mul(self, _: tarpc::context::Context, modulus: u64) -> FheResult {
        FheResult {
            status: if self.primal.device().is_some() {
                "accepted"
            } else {
                "no_device"
            }
            .to_string(),
            modulus,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_serialization() {
        let info = DeviceInfo {
            name: "Test GPU".to_string(),
            vendor: 1234,
            device_type: "DiscreteGpu".to_string(),
            backend: "Vulkan".to_string(),
            driver: "test".to_string(),
            driver_info: "1.0".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("Test GPU"));
    }

    #[test]
    fn test_tolerances() {
        let tol = Tolerances {
            name: "fhe".to_string(),
            abs_tol: 0.0,
            rel_tol: 0.0,
        };
        let json = serde_json::to_string(&tol).unwrap();
        let parsed: Tolerances = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "fhe");
        assert_eq!(parsed.abs_tol, 0.0);
    }
}
