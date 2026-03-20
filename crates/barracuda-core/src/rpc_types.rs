// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared types for the barraCuda RPC interface.
//!
//! Used by both the tarpc service and JSON-RPC method handlers. Kept in a
//! separate module so consumers can import types without pulling in the
//! full tarpc dependency.
//!
//! String fields use `String` (not `&str`) because these types cross RPC
//! boundaries via serde — owned strings are required for deserialization.

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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatmulResult {
    /// Status of the operation.
    pub status: String,
    /// Result tensor ID.
    pub result_id: String,
    /// Shape of the result tensor.
    pub shape: Vec<usize>,
}

/// Primal identity for runtime capability-based discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalInfo {
    /// Primal name.
    pub primal: String,
    /// Primal version.
    pub version: String,
    /// IPC protocol.
    pub protocol: String,
    /// Method namespace.
    pub namespace: String,
    /// License identifier.
    pub license: String,
}

/// Capability advertisement for runtime discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalCapabilities {
    /// Domain capabilities (gpu_compute, tensor_ops, fhe, etc.).
    pub domains: Vec<String>,
    /// Available JSON-RPC methods.
    pub methods: Vec<String>,
    /// Whether a GPU is available.
    pub gpu_available: bool,
    /// Whether f64 shaders work.
    pub f64_shaders: bool,
    /// Whether SPIR-V passthrough is available.
    pub spirv_passthrough: bool,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn device_info_roundtrip() {
        let info = DeviceInfo {
            name: "Test GPU".into(),
            vendor: 0x10DE,
            device_type: "DiscreteGpu".into(),
            backend: "Vulkan".into(),
            driver: "nvidia".into(),
            driver_info: "550.0".into(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: DeviceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "Test GPU");
        assert_eq!(parsed.vendor, 0x10DE);
    }

    #[test]
    fn device_probe_roundtrip() {
        let probe = DeviceProbe {
            available: true,
            max_buffer_size: 2_147_483_648,
            max_storage_buffers: 8,
            max_workgroup_size_x: 256,
            max_workgroups_per_dimension: 65_535,
        };
        let json = serde_json::to_string(&probe).unwrap();
        let parsed: DeviceProbe = serde_json::from_str(&json).unwrap();
        assert!(parsed.available);
        assert_eq!(parsed.max_buffer_size, 2_147_483_648);
    }

    #[test]
    fn health_report_roundtrip() {
        let report = HealthReport {
            name: "barraCuda".into(),
            version: "0.3.5".into(),
            status: "healthy".into(),
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: HealthReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "barraCuda");
        assert_eq!(parsed.version, "0.3.5");
    }

    #[test]
    fn tolerances_roundtrip() {
        let tol = Tolerances {
            name: "GPU_RELAXED".into(),
            abs_tol: 1e-4,
            rel_tol: 1e-3,
        };
        let json = serde_json::to_string(&tol).unwrap();
        let parsed: Tolerances = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "GPU_RELAXED");
        assert!((parsed.abs_tol - 1e-4).abs() < f64::EPSILON);
    }

    #[test]
    fn tensor_handle_roundtrip() {
        let handle = TensorHandle {
            tensor_id: "t_001".into(),
            shape: vec![2, 3, 4],
            elements: 24,
            dtype: "f32".into(),
        };
        let json = serde_json::to_string(&handle).unwrap();
        let parsed: TensorHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.shape, vec![2, 3, 4]);
        assert_eq!(parsed.elements, 24);
    }

    #[test]
    fn primal_info_roundtrip() {
        let info = PrimalInfo {
            primal: "barraCuda".into(),
            version: "0.3.5".into(),
            protocol: "jsonrpc-2.0".into(),
            namespace: "barracuda".into(),
            license: "AGPL-3.0-or-later".into(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: PrimalInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.primal, "barraCuda");
        assert_eq!(parsed.license, "AGPL-3.0-or-later");
    }

    #[test]
    fn primal_capabilities_roundtrip() {
        let caps = PrimalCapabilities {
            domains: vec!["gpu_compute".into(), "tensor_ops".into()],
            methods: vec!["barracuda.device.list".into()],
            gpu_available: true,
            f64_shaders: false,
            spirv_passthrough: false,
        };
        let json = serde_json::to_string(&caps).unwrap();
        let parsed: PrimalCapabilities = serde_json::from_str(&json).unwrap();
        assert!(parsed.gpu_available);
        assert_eq!(parsed.domains.len(), 2);
    }

    #[test]
    fn dispatch_result_optional_fields() {
        let result = DispatchResult {
            status: "ok".into(),
            tensor_id: None,
            shape: None,
            data: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("tensor_id") || json.contains("null"));
        let parsed: DispatchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.status, "ok");
    }

    #[test]
    fn fhe_ntt_result_roundtrip() {
        let result = FheNttResult {
            status: "ok".into(),
            result: vec![1, 2, 3, 4],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: FheNttResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn matmul_result_roundtrip() {
        let result = MatmulResult {
            status: "ok".into(),
            result_id: "matmul_001".into(),
            shape: vec![4, 4],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: MatmulResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, result);
    }
}
