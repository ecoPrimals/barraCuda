// SPDX-License-Identifier: AGPL-3.0-only
//! tarpc service definition for barraCuda.
//!
//! High-performance binary RPC alongside the JSON-RPC 2.0 text protocol.
//! Per wateringHole `UNIVERSAL_IPC_STANDARD_V3.md`, tarpc is the optional
//! high-throughput protocol for primal-to-primal calls. JSON-RPC 2.0 remains
//! the primary protocol for external/cross-language consumers.
//!
//! ## String parameters
//!
//! Service methods use `String` (not `&str`) for string parameters because
//! tarpc uses serde for serialization across process/network boundaries.
//! `&str` cannot be deserialized into (it doesn't own data); `impl Into<String>`
//! is not `Serialize`; `Cow<'static, str>` adds complexity without benefit
//! over the wire. For RPC boundaries, `String` is the idiomatic type — `&str`
//! and `Cow` are for in-process APIs.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda_core::rpc::BarraCudaClient;
//!
//! let client = BarraCudaClient::new(/* ... */);
//! let devices = client.device_list(context::current()).await?;
//! ```

pub use crate::rpc_types::*;

/// barraCuda tarpc service.
///
/// Mirrors the JSON-RPC 2.0 endpoints with strongly-typed signatures.
/// All methods follow the semantic naming standard: `barracuda.{domain}.{operation}`.
#[tarpc::service]
pub trait BarraCudaService {
    /// Primal identity — for runtime discovery, not hardcoded names.
    async fn primal_info() -> PrimalInfo;

    /// Advertise capabilities for capability-based routing.
    async fn primal_capabilities() -> PrimalCapabilities;

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

/// Pack u64 coefficients into a wgpu-compatible tensor (u64 → split u32 pairs → f32 bits).
fn u64_to_tensor(
    coeffs: &[u64],
    dev: &barracuda::device::WgpuDevice,
) -> std::result::Result<barracuda::tensor::Tensor, barracuda::error::BarracudaError> {
    let f32_bits: Vec<f32> = coeffs
        .iter()
        .flat_map(|&x| [f32::from_bits(x as u32), f32::from_bits((x >> 32) as u32)])
        .collect();
    barracuda::tensor::Tensor::from_data(
        &f32_bits,
        vec![coeffs.len() * 2],
        std::sync::Arc::new(dev.clone()),
    )
}

/// Unpack u32 pairs back into u64 coefficients.
fn u32_pairs_to_u64(data: &[u32]) -> Vec<u64> {
    data.chunks(2)
        .map(|c| u64::from(c[0]) | (u64::from(c.get(1).copied().unwrap_or(0)) << 32))
        .collect()
}

impl BarraCudaService for BarraCudaServer {
    async fn primal_info(self, _: tarpc::context::Context) -> PrimalInfo {
        PrimalInfo {
            primal: crate::PRIMAL_NAME.into(),
            version: env!("CARGO_PKG_VERSION").into(),
            protocol: "json-rpc-2.0".into(),
            namespace: crate::PRIMAL_NAMESPACE.into(),
            license: "AGPL-3.0-only".into(),
        }
    }

    async fn primal_capabilities(self, _: tarpc::context::Context) -> PrimalCapabilities {
        let dev = self.primal.device();
        let has_gpu = dev.is_some();
        let has_f64 = dev.as_ref().is_some_and(|d| d.has_f64_shaders());
        let has_spirv = dev.as_ref().is_some_and(|d| d.has_spirv_passthrough());

        PrimalCapabilities {
            domains: vec![
                "gpu_compute".into(),
                "tensor_ops".into(),
                "fhe".into(),
                "molecular_dynamics".into(),
                "lattice_qcd".into(),
                "statistics".into(),
                "hydrology".into(),
                "bio".into(),
            ],
            methods: crate::ipc::methods::REGISTERED_METHODS
                .iter()
                .map(|&m| m.into())
                .collect(),
            gpu_available: has_gpu,
            f64_shaders: has_f64,
            spirv_passthrough: has_spirv,
        }
    }

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
        let Some(dev) = self.primal.device() else {
            return DeviceProbe {
                available: false,
                max_buffer_size: 0,
                max_storage_buffers: 0,
                max_workgroup_size_x: 0,
                max_workgroups_per_dimension: 0,
            };
        };
        let limits = dev.device().limits();
        DeviceProbe {
            available: true,
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffers: limits.max_storage_buffers_per_shader_stage,
            max_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    async fn health_check(self, _: tarpc::context::Context) -> HealthReport {
        use crate::health::PrimalHealth;
        match self.primal.health_check().await {
            Ok(report) => HealthReport {
                name: report.name,
                version: report.version,
                status: report.status.to_string(),
            },
            Err(e) => HealthReport {
                name: crate::PRIMAL_NAME.to_string(),
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

    /// Run GPU stack validation via matmul identity test.
    /// Uses 2×2 identity matrix: minimal size that validates matmul path without
    /// unnecessary GPU memory/transfer overhead.
    async fn validate_gpu_stack(self, _: tarpc::context::Context) -> ValidationResult {
        let Some(dev) = self.primal.device() else {
            return ValidationResult {
                gpu_available: false,
                status: "no_device".to_string(),
                message: "No GPU device available".to_string(),
            };
        };

        let dev_arc = std::sync::Arc::new(dev);
        let matmul_pass = {
            let eye = vec![1.0, 0.0, 0.0, 1.0];
            let inp = vec![1.0, 2.0, 3.0, 4.0];
            barracuda::tensor::Tensor::from_data(&eye, vec![2, 2], dev_arc.clone())
                .and_then(|e| {
                    let i =
                        barracuda::tensor::Tensor::from_data(&inp, vec![2, 2], dev_arc.clone())?;
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

        let dev_arc = std::sync::Arc::new(dev);
        match op.as_str() {
            "zeros" | "ones" => {
                let s = shape.unwrap_or_else(|| vec![1]);
                let elements: usize = s.iter().product();
                let fill = if op == "ones" { 1.0f32 } else { 0.0f32 };
                let values = vec![fill; elements];
                match barracuda::tensor::Tensor::from_data(&values, s.clone(), dev_arc) {
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
        match barracuda::tensor::Tensor::from_data(&values, shape.clone(), std::sync::Arc::new(dev))
        {
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
                status: "no_device".into(),
                result: vec![],
            };
        };

        let tensor = match u64_to_tensor(&coefficients, &dev) {
            Ok(t) => t,
            Err(e) => {
                return FheNttResult {
                    status: format!("error: {e}"),
                    result: vec![],
                };
            }
        };

        let Ok(degree_u32) = u32::try_from(degree) else {
            return FheNttResult {
                status: format!("error: degree {degree} exceeds u32::MAX"),
                result: vec![],
            };
        };

        let ntt = match barracuda::ops::fhe_ntt::FheNtt::new(
            tensor,
            degree_u32,
            modulus,
            root_of_unity,
        ) {
            Ok(n) => n,
            Err(e) => {
                return FheNttResult {
                    status: format!("error: {e}"),
                    result: vec![],
                };
            }
        };

        match ntt.execute().and_then(|t| t.to_vec_u32()) {
            Ok(u32_data) => FheNttResult {
                status: "completed".into(),
                result: u32_pairs_to_u64(&u32_data),
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
                status: "no_device".into(),
                result: vec![],
            };
        };

        let tensor_a = match u64_to_tensor(&a, &dev) {
            Ok(t) => t,
            Err(e) => {
                return FhePointwiseMulResult {
                    status: format!("error: {e}"),
                    result: vec![],
                };
            }
        };
        let tensor_b = match u64_to_tensor(&b, &dev) {
            Ok(t) => t,
            Err(e) => {
                return FhePointwiseMulResult {
                    status: format!("error: {e}"),
                    result: vec![],
                };
            }
        };

        let Ok(degree_u32) = u32::try_from(degree) else {
            return FhePointwiseMulResult {
                status: format!("error: degree {degree} exceeds u32::MAX"),
                result: vec![],
            };
        };

        let mul = match barracuda::ops::fhe_pointwise_mul::FhePointwiseMul::new(
            tensor_a, tensor_b, degree_u32, modulus,
        ) {
            Ok(m) => m,
            Err(e) => {
                return FhePointwiseMulResult {
                    status: format!("error: {e}"),
                    result: vec![],
                };
            }
        };

        match mul.execute().and_then(|t| t.to_vec_u32()) {
            Ok(u32_data) => FhePointwiseMulResult {
                status: "completed".into(),
                result: u32_pairs_to_u64(&u32_data),
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
