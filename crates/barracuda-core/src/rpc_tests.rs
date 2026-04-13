// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for the tarpc service implementation.

#![expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]

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
#[expect(clippy::float_cmp, reason = "exact tolerance comparison in test")]
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

#[test]
fn u32_pairs_to_u64_roundtrip() {
    let original: Vec<u64> = vec![0, 1, u64::MAX, 0x0000_0001_0000_0002];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentional u64→u32 split mirroring production u64_to_tensor"
    )]
    let u32_data: Vec<u32> = original
        .iter()
        .flat_map(|&x| [x as u32, (x >> 32) as u32])
        .collect();
    let recovered = u32_pairs_to_u64(&u32_data);
    assert_eq!(recovered, original);
}

#[test]
fn u32_pairs_to_u64_empty() {
    let result = u32_pairs_to_u64(&[]);
    assert!(result.is_empty());
}

#[test]
fn u32_pairs_to_u64_odd_length() {
    let result = u32_pairs_to_u64(&[42]);
    assert_eq!(result, vec![42]);
}

#[test]
fn server_construction() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let _server = BarraCudaServer::new(primal);
}

#[tokio::test]
async fn tarpc_primal_info() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let info = BarraCudaService::primal_info(server, tarpc::context::current()).await;
    assert_eq!(info.primal, "barraCuda");
    assert_eq!(info.protocol, "json-rpc-2.0");
    assert_eq!(info.namespace, "barracuda");
    assert_eq!(info.license, "AGPL-3.0-or-later");
}

#[tokio::test]
async fn tarpc_primal_capabilities_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let caps = BarraCudaService::primal_capabilities(server, tarpc::context::current()).await;
    assert!(!caps.gpu_available);
    assert!(!caps.f64_shaders);
    assert!(!caps.spirv_passthrough);
    assert!(!caps.domains.is_empty());
    assert!(!caps.methods.is_empty());
}

#[tokio::test]
async fn tarpc_device_list_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let devices = BarraCudaService::device_list(server, tarpc::context::current()).await;
    assert!(devices.is_empty());
}

#[tokio::test]
async fn tarpc_device_probe_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let probe = BarraCudaService::device_probe(server, tarpc::context::current()).await;
    assert!(!probe.available);
    assert_eq!(probe.max_buffer_size, 0);
}

#[tokio::test]
async fn tarpc_health_liveness() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let report = BarraCudaService::health_liveness(server, tarpc::context::current()).await;
    assert_eq!(report.status, "alive");
}

#[tokio::test]
async fn tarpc_health_readiness() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let report = BarraCudaService::health_readiness(server, tarpc::context::current()).await;
    assert_eq!(report.status, "not_ready");
    assert!(!report.gpu_available);
}

#[tokio::test]
async fn tarpc_health_check() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let report = BarraCudaService::health_check(server, tarpc::context::current()).await;
    assert_eq!(report.name, "barraCuda");
}

#[tokio::test]
async fn tarpc_tolerances_get_fhe() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol =
        BarraCudaService::tolerances_get(server, tarpc::context::current(), "fhe".to_string())
            .await;
    assert_eq!(tol.name, "fhe");
    assert!(
        tol.abs_tol <= f64::EPSILON,
        "FHE maps to DETERMINISM tier (near-zero abs_tol)"
    );
}

#[tokio::test]
#[expect(
    clippy::float_cmp,
    reason = "comparing exact constants from tolerance registry"
)]
async fn tarpc_tolerances_get_f64() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol =
        BarraCudaService::tolerances_get(server, tarpc::context::current(), "f64".to_string())
            .await;
    assert_eq!(tol.name, "f64");
    let acc = barracuda::tolerances::ACCUMULATION;
    assert_eq!(tol.abs_tol, acc.abs_tol);
    assert_eq!(tol.rel_tol, acc.rel_tol);
}

#[tokio::test]
#[expect(
    clippy::float_cmp,
    reason = "comparing exact constants from tolerance registry"
)]
async fn tarpc_tolerances_get_by_name() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol = BarraCudaService::tolerances_get(
        server,
        tarpc::context::current(),
        "pharma_foce".to_string(),
    )
    .await;
    assert_eq!(tol.name, "pharma_foce");
    let expected = barracuda::tolerances::PHARMA_FOCE;
    assert_eq!(tol.abs_tol, expected.abs_tol);
}

#[tokio::test]
#[expect(
    clippy::float_cmp,
    reason = "comparing exact constants from tolerance registry"
)]
async fn tarpc_tolerances_get_by_tier() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol = BarraCudaService::tolerances_get(
        server,
        tarpc::context::current(),
        "transcendental".to_string(),
    )
    .await;
    assert_eq!(tol.name, "transcendental");
    let expected = barracuda::tolerances::TRANSCENDENTAL;
    assert_eq!(tol.abs_tol, expected.abs_tol);
}

#[tokio::test]
async fn tarpc_tolerances_get_unknown() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol =
        BarraCudaService::tolerances_get(server, tarpc::context::current(), "whatever".to_string())
            .await;
    assert!(tol.abs_tol > 0.0);
}

#[tokio::test]
async fn tarpc_validate_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::validate_gpu_stack(server, tarpc::context::current()).await;
    assert!(!result.gpu_available);
    assert_eq!(result.status, "no_device");
}

#[tokio::test]
async fn tarpc_compute_dispatch_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::compute_dispatch(
        server,
        tarpc::context::current(),
        "zeros".to_string(),
        Some(vec![4]),
        None,
    )
    .await;
    assert_eq!(result.status, "no_device");
}

#[tokio::test]
async fn tarpc_tensor_create_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let handle = BarraCudaService::tensor_create(
        server,
        tarpc::context::current(),
        vec![2, 3],
        "f32".to_string(),
        None,
    )
    .await;
    assert!(handle.tensor_id.is_empty());
}

#[tokio::test]
async fn tarpc_tensor_matmul_no_tensors() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::tensor_matmul(
        server,
        tarpc::context::current(),
        "nonexistent_a".to_string(),
        "nonexistent_b".to_string(),
    )
    .await;
    assert_eq!(result.status, "tensor_not_found");
}

#[tokio::test]
async fn tarpc_fhe_ntt_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::fhe_ntt(
        server,
        tarpc::context::current(),
        17,
        4,
        4,
        vec![1, 2, 3, 4],
    )
    .await;
    assert_eq!(result.status, "no_device");
}

#[tokio::test]
async fn tarpc_fhe_pointwise_mul_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::fhe_pointwise_mul(
        server,
        tarpc::context::current(),
        17,
        4,
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
    )
    .await;
    assert_eq!(result.status, "no_device");
}

#[tokio::test]
async fn tarpc_compute_dispatch_read_nonexistent() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::compute_dispatch(
        server,
        tarpc::context::current(),
        "read".to_string(),
        None,
        Some("nonexistent".to_string()),
    )
    .await;
    assert_eq!(result.status, "tensor_not_found");
}

#[tokio::test]
async fn tarpc_compute_dispatch_unknown_op() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::compute_dispatch(
        server,
        tarpc::context::current(),
        "unknown_op".to_string(),
        None,
        None,
    )
    .await;
    assert_eq!(result.status, "unknown_op");
}

#[tokio::test]
async fn tarpc_compute_dispatch_ones_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::compute_dispatch(
        server,
        tarpc::context::current(),
        "ones".to_string(),
        Some(vec![2, 3]),
        None,
    )
    .await;
    assert_eq!(result.status, "no_device");
}

#[tokio::test]
async fn tarpc_compute_dispatch_zeros_default_shape() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::compute_dispatch(
        server,
        tarpc::context::current(),
        "zeros".to_string(),
        None,
        None,
    )
    .await;
    assert_eq!(result.status, "no_device");
}

#[tokio::test]
async fn tarpc_compute_dispatch_read_empty_tensor_id() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::compute_dispatch(
        server,
        tarpc::context::current(),
        "read".to_string(),
        None,
        None,
    )
    .await;
    assert_eq!(result.status, "tensor_not_found");
}

// ── tarpc tensor_create branch coverage ──────────────────────────────

#[tokio::test]
async fn tarpc_tensor_create_with_data_no_gpu() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let handle = BarraCudaService::tensor_create(
        server,
        tarpc::context::current(),
        vec![2, 2],
        "f32".to_string(),
        Some(vec![1.0, 2.0, 3.0, 4.0]),
    )
    .await;
    assert!(handle.tensor_id.is_empty());
    assert_eq!(handle.elements, 4);
    assert_eq!(handle.shape, vec![2, 2]);
}

// ── tarpc tolerances branches ────────────────────────────────────────

#[tokio::test]
async fn tarpc_tolerances_get_df64() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol =
        BarraCudaService::tolerances_get(server, tarpc::context::current(), "df64".to_string())
            .await;
    assert_eq!(tol.name, "df64");
    assert!(tol.abs_tol > 0.0);
}

#[tokio::test]
async fn tarpc_tolerances_get_double() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol =
        BarraCudaService::tolerances_get(server, tarpc::context::current(), "double".to_string())
            .await;
    assert_eq!(tol.name, "double");
}

#[tokio::test]
async fn tarpc_tolerances_get_emulated_double() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let tol = BarraCudaService::tolerances_get(
        server,
        tarpc::context::current(),
        "emulated_double".to_string(),
    )
    .await;
    assert_eq!(tol.name, "emulated_double");
}

#[tokio::test]
async fn tarpc_identity_get() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let info = BarraCudaService::identity_get(server, tarpc::context::current()).await;
    assert_eq!(info.primal, crate::PRIMAL_NAMESPACE);
    assert_eq!(info.domain, "math");
    assert_eq!(info.license, "AGPL-3.0-or-later");
    assert!(!info.version.is_empty());
}

#[tokio::test]
async fn tarpc_fhe_ntt_degree_overflow() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::fhe_ntt(
        server,
        tarpc::context::current(),
        17,
        u64::from(u32::MAX) + 1,
        3,
        vec![1, 2, 3],
    )
    .await;
    assert!(
        result.status.contains("error"),
        "degree exceeding u32::MAX should produce error status"
    );
    assert!(result.result.is_empty());
}

#[tokio::test]
async fn tarpc_fhe_pointwise_mul_degree_overflow() {
    let primal = std::sync::Arc::new(crate::BarraCudaPrimal::new());
    let server = BarraCudaServer::new(primal);
    let result = BarraCudaService::fhe_pointwise_mul(
        server,
        tarpc::context::current(),
        17,
        u64::from(u32::MAX) + 1,
        vec![1],
        vec![2],
    )
    .await;
    assert!(
        result.status.contains("error"),
        "degree exceeding u32::MAX should produce error status"
    );
    assert!(result.result.is_empty());
}

#[tokio::test]
async fn tarpc_health_readiness_after_start() {
    let mut primal = crate::BarraCudaPrimal::new();
    let _ = crate::lifecycle::PrimalLifecycle::start(&mut primal).await;
    let primal = std::sync::Arc::new(primal);
    let server = BarraCudaServer::new(primal);
    let report = BarraCudaService::health_readiness(server, tarpc::context::current()).await;
    assert_eq!(report.status, "ready");
}
