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
    /// Tensor ID (if operation created/returned a tensor).
    pub tensor_id: Option<String>,
    /// Shape of the result tensor (if applicable).
    pub shape: Option<Vec<usize>>,
    /// Read-back data (for "read" operations).
    pub data: Option<Vec<f32>>,
}

/// FHE NTT result with transformed coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FheNttResult {
    /// Status of the operation.
    pub status: String,
    /// Transformed coefficients.
    pub result: Vec<u64>,
}

/// FHE pointwise multiplication result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhePointwiseMulResult {
    /// Status of the operation.
    pub status: String,
    /// Product coefficients.
    pub result: Vec<u64>,
}

/// Matmul result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulResult {
    /// Status of the operation.
    pub status: String,
    /// Result tensor ID.
    pub result_id: String,
    /// Shape of the result tensor.
    pub shape: Vec<usize>,
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

    /// Dispatch a named compute operation (zeros, ones, read).
    async fn compute_dispatch(
        op: String,
        shape: Option<Vec<usize>>,
        tensor_id: Option<String>,
    ) -> DispatchResult;

    /// Create a tensor on the device with optional initial data.
    async fn tensor_create(
        shape: Vec<usize>,
        dtype: String,
        data: Option<Vec<f32>>,
    ) -> TensorHandle;

    /// Matrix multiply two tensors by ID.
    async fn tensor_matmul(lhs_id: String, rhs_id: String) -> MatmulResult;

    /// FHE Number Theoretic Transform.
    async fn fhe_ntt(
        modulus: u64,
        degree: u64,
        root_of_unity: u64,
        coefficients: Vec<u64>,
    ) -> FheNttResult;

    /// FHE pointwise polynomial multiplication.
    async fn fhe_pointwise_mul(
        modulus: u64,
        degree: u64,
        a: Vec<u64>,
        b: Vec<u64>,
    ) -> FhePointwiseMulResult;
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
        use crate::health::PrimalHealth;
        match self.primal.health_check().await {
            Ok(report) => HealthReport {
                name: report.name.clone(),
                version: report.version.clone(),
                status: format!("{}", report.status),
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
        let Some(dev) = self.primal.device() else {
            return ValidationResult {
                gpu_available: false,
                status: "no_device".to_string(),
                message: "No GPU device available".to_string(),
            };
        };

        let matmul_pass = {
            let eye = vec![1.0, 0.0, 0.0, 1.0];
            let inp = vec![1.0, 2.0, 3.0, 4.0];
            barracuda::tensor::Tensor::from_data(&eye, vec![2, 2], dev.clone())
                .and_then(|e| {
                    let i = barracuda::tensor::Tensor::from_data(&inp, vec![2, 2], dev.clone())?;
                    i.matmul(&e)
                })
                .and_then(|r| r.to_vec())
                .map(|v| v.iter().zip(&inp).all(|(a, b)| (a - b).abs() < 1e-4))
                .unwrap_or(false)
        };

        ValidationResult {
            gpu_available: true,
            status: if matmul_pass { "pass" } else { "partial_fail" }.to_string(),
            message: format!(
                "matmul_identity: {}",
                if matmul_pass { "pass" } else { "fail" }
            ),
        }
    }

    async fn compute_dispatch(
        self,
        _: tarpc::context::Context,
        op: String,
        shape: Option<Vec<usize>>,
        tensor_id: Option<String>,
    ) -> DispatchResult {
        let Some(dev) = self.primal.device() else {
            return DispatchResult {
                status: "no_device".to_string(),
                tensor_id: None,
                shape: None,
                data: None,
            };
        };

        match op.as_str() {
            "zeros" | "ones" => {
                let s = shape.unwrap_or_else(|| vec![1]);
                let elements: usize = s.iter().product();
                let fill = if op == "ones" { 1.0f32 } else { 0.0f32 };
                let values = vec![fill; elements];
                match barracuda::tensor::Tensor::from_data(&values, s.clone(), dev.clone()) {
                    Ok(tensor) => {
                        let tid = self.primal.store_tensor(tensor);
                        DispatchResult {
                            status: "completed".to_string(),
                            tensor_id: Some(tid),
                            shape: Some(s),
                            data: None,
                        }
                    }
                    Err(e) => DispatchResult {
                        status: format!("error: {e}"),
                        tensor_id: None,
                        shape: Some(s),
                        data: None,
                    },
                }
            }
            "read" => {
                let tid = tensor_id.unwrap_or_default();
                match self.primal.get_tensor(&tid) {
                    Some(tensor) => match tensor.to_vec() {
                        Ok(values) => DispatchResult {
                            status: "completed".to_string(),
                            tensor_id: Some(tid),
                            shape: Some(tensor.shape().to_vec()),
                            data: Some(values),
                        },
                        Err(e) => DispatchResult {
                            status: format!("error: {e}"),
                            tensor_id: Some(tid),
                            shape: None,
                            data: None,
                        },
                    },
                    None => DispatchResult {
                        status: "tensor_not_found".to_string(),
                        tensor_id: Some(tid),
                        shape: None,
                        data: None,
                    },
                }
            }
            _ => DispatchResult {
                status: "unknown_op".to_string(),
                tensor_id: None,
                shape: None,
                data: None,
            },
        }
    }

    async fn tensor_create(
        self,
        _: tarpc::context::Context,
        shape: Vec<usize>,
        dtype: String,
        data: Option<Vec<f32>>,
    ) -> TensorHandle {
        let elements: usize = shape.iter().product();

        let Some(dev) = self.primal.device() else {
            return TensorHandle {
                tensor_id: String::new(),
                shape,
                elements,
                dtype,
            };
        };

        let values = data.unwrap_or_else(|| vec![0.0f32; elements]);
        match barracuda::tensor::Tensor::from_data(&values, shape.clone(), dev.clone()) {
            Ok(tensor) => {
                let tensor_id = self.primal.store_tensor(tensor);
                TensorHandle {
                    tensor_id,
                    shape,
                    elements,
                    dtype,
                }
            }
            Err(_) => TensorHandle {
                tensor_id: String::new(),
                shape,
                elements,
                dtype,
            },
        }
    }

    async fn tensor_matmul(
        self,
        _: tarpc::context::Context,
        lhs_id: String,
        rhs_id: String,
    ) -> MatmulResult {
        let lhs = self.primal.get_tensor(&lhs_id);
        let rhs = self.primal.get_tensor(&rhs_id);

        match (lhs, rhs) {
            (Some(l), Some(r)) => match l.matmul_ref(&r) {
                Ok(result) => {
                    let result_shape = result.shape().to_vec();
                    let result_id = self.primal.store_tensor(result);
                    MatmulResult {
                        status: "completed".to_string(),
                        result_id,
                        shape: result_shape,
                    }
                }
                Err(e) => MatmulResult {
                    status: format!("error: {e}"),
                    result_id: String::new(),
                    shape: vec![],
                },
            },
            _ => MatmulResult {
                status: "tensor_not_found".to_string(),
                result_id: String::new(),
                shape: vec![],
            },
        }
    }

    async fn fhe_ntt(
        self,
        _: tarpc::context::Context,
        modulus: u64,
        degree: u64,
        root_of_unity: u64,
        coefficients: Vec<u64>,
    ) -> FheNttResult {
        let Some(dev) = self.primal.device() else {
            return FheNttResult {
                status: "no_device".to_string(),
                result: vec![],
            };
        };

        let u32_pairs: Vec<u32> = coefficients
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();

        let tensor = match barracuda::tensor::Tensor::from_data(
            &f32_bits,
            vec![coefficients.len() * 2],
            dev.clone(),
        ) {
            Ok(t) => t,
            Err(e) => {
                return FheNttResult {
                    status: format!("error: {e}"),
                    result: vec![],
                }
            }
        };

        let ntt = match barracuda::ops::fhe_ntt::FheNtt::new(
            tensor,
            degree as u32,
            modulus,
            root_of_unity,
        ) {
            Ok(n) => n,
            Err(e) => {
                return FheNttResult {
                    status: format!("error: {e}"),
                    result: vec![],
                }
            }
        };

        match ntt.execute() {
            Ok(result_tensor) => match result_tensor.to_vec_u32() {
                Ok(u32_data) => {
                    let result_u64: Vec<u64> = u32_data
                        .chunks(2)
                        .map(|c| {
                            u64::from(c[0]) | (u64::from(c.get(1).copied().unwrap_or(0)) << 32)
                        })
                        .collect();
                    FheNttResult {
                        status: "completed".to_string(),
                        result: result_u64,
                    }
                }
                Err(e) => FheNttResult {
                    status: format!("error: {e}"),
                    result: vec![],
                },
            },
            Err(e) => FheNttResult {
                status: format!("error: {e}"),
                result: vec![],
            },
        }
    }

    async fn fhe_pointwise_mul(
        self,
        _: tarpc::context::Context,
        modulus: u64,
        degree: u64,
        a: Vec<u64>,
        b: Vec<u64>,
    ) -> FhePointwiseMulResult {
        let Some(dev) = self.primal.device() else {
            return FhePointwiseMulResult {
                status: "no_device".to_string(),
                result: vec![],
            };
        };

        let convert_to_tensor =
            |poly: &[u64], label: &str| -> std::result::Result<barracuda::tensor::Tensor, String> {
                let u32_pairs: Vec<u32> = poly
                    .iter()
                    .flat_map(|&x| [x as u32, (x >> 32) as u32])
                    .collect();
                let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();
                barracuda::tensor::Tensor::from_data(&f32_bits, vec![poly.len() * 2], dev.clone())
                    .map_err(|e| format!("{label}: {e}"))
            };

        let tensor_a = match convert_to_tensor(&a, "input_a") {
            Ok(t) => t,
            Err(e) => {
                return FhePointwiseMulResult {
                    status: format!("error: {e}"),
                    result: vec![],
                }
            }
        };

        let tensor_b = match convert_to_tensor(&b, "input_b") {
            Ok(t) => t,
            Err(e) => {
                return FhePointwiseMulResult {
                    status: format!("error: {e}"),
                    result: vec![],
                }
            }
        };

        let mul = match barracuda::ops::fhe_pointwise_mul::FhePointwiseMul::new(
            tensor_a,
            tensor_b,
            degree as u32,
            modulus,
        ) {
            Ok(m) => m,
            Err(e) => {
                return FhePointwiseMulResult {
                    status: format!("error: {e}"),
                    result: vec![],
                }
            }
        };

        match mul.execute() {
            Ok(result_tensor) => match result_tensor.to_vec_u32() {
                Ok(u32_data) => {
                    let result_u64: Vec<u64> = u32_data
                        .chunks(2)
                        .map(|c| {
                            u64::from(c[0]) | (u64::from(c.get(1).copied().unwrap_or(0)) << 32)
                        })
                        .collect();
                    FhePointwiseMulResult {
                        status: "completed".to_string(),
                        result: result_u64,
                    }
                }
                Err(e) => FhePointwiseMulResult {
                    status: format!("error: {e}"),
                    result: vec![],
                },
            },
            Err(e) => FhePointwiseMulResult {
                status: format!("error: {e}"),
                result: vec![],
            },
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "suppressed")]
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
    #[expect(clippy::float_cmp, reason = "tests")]
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
