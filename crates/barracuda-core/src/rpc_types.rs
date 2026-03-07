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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
