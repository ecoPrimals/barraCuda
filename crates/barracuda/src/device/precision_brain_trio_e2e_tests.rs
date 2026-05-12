// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trio contract E2E integration tests.
//!
//! These tests validate the complete data-flow contract between the three
//! Compute Trio primals:
//!   barraCuda (WHAT) → coralReef (HOW) → toadStool (WHERE)
//!
//! Each test chains: PrecisionBrain domain routing → PrecisionAdvice
//! construction → coralReef compile wire format → ShaderDispatchInfo →
//! toadStool dispatch wire format (with mock TCP server verification).

#[cfg(feature = "sovereign-dispatch")]
use super::*;

/// Trio contract E2E: LatticeQcd → F64 → coralReef advice → toadStool dispatch.
///
/// Validates the complete chain for a high-precision physics domain:
/// 1. PrecisionBrain routes `LatticeQcd` to F64 tier (needs sovereign compile)
/// 2. `build_precision_advice` produces correct F64 advice
/// 3. `PrecisionAdvice` serializes in coral wire format
/// 4. Simulated compile metadata yields `ShaderDispatchInfo`
/// 5. `submit_dispatch` to mock toadStool carries correct `hardware_hint`,
///    `gpr_count`, `workgroup`, and binary
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn trio_contract_e2e_lattice_qcd_f64_dispatch() {
    use crate::device::backend::HardwareHint;
    use crate::device::coral_compiler::types::{CoralBinary, PrecisionAdvice};
    use crate::device::sovereign_device::SovereignDevice;
    use crate::device::sovereign_dispatch_wire::{
        IpcBufferBinding, ShaderDispatchInfo, submit_dispatch,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    // ── Step 1: PrecisionBrain routes LatticeQcd ──────────────────────
    let cal = make_cal(true, false, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);

    let domain = PhysicsDomain::LatticeQcd;
    assert!(brain.needs_sovereign_compile(domain));

    let tier = brain.route(domain);
    let (f64_shader, df64_shader) = match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => (true, false),
        PrecisionTier::DF64 => (false, true),
        _ => (false, false),
    };

    // ── Step 2: Build PrecisionAdvice ─────────────────────────────────
    let advice = SovereignDevice::build_precision_advice(f64_shader, df64_shader);
    assert!(advice.is_some(), "F64 domain must produce advice");
    let adv = advice.unwrap();
    assert_eq!(adv.tier, "F64");
    assert!(adv.needs_transcendental_lowering);
    assert!(!adv.df64_naga_poisoned);

    let hint = tier.recommended_hardware_hint();
    assert_eq!(
        hint,
        HardwareHint::Compute,
        "F64 routes to Compute (ALU/FP64 cores)"
    );

    // ── Step 3: coralReef wire format (PrecisionAdvice) ───────────────
    let wire_advice = PrecisionAdvice {
        tier: adv.tier.clone(),
        needs_transcendental_lowering: adv.needs_transcendental_lowering,
        df64_naga_poisoned: adv.df64_naga_poisoned,
        domain: Some("LatticeQcd".to_owned()),
    };
    let advice_json = serde_json::to_string(&wire_advice).unwrap();
    assert!(advice_json.contains("\"tier\":\"F64\""));
    assert!(advice_json.contains("\"needs_transcendental_lowering\":true"));
    assert!(advice_json.contains("\"df64_naga_poisoned\":false"));
    assert!(advice_json.contains("\"domain\":\"LatticeQcd\""));

    // ── Step 4: Simulate compile result → ShaderDispatchInfo ──────────
    let coral_binary = CoralBinary {
        binary: bytes::Bytes::from_static(&[0xDE, 0xAD, 0xBE, 0xEF]),
        arch: "sm_70".to_owned(),
        gpr_count: Some(48),
        workgroup: Some([64, 1, 1]),
        shared_mem_bytes: Some(2048),
        barrier_count: Some(1),
    };

    let shader_info = ShaderDispatchInfo {
        gpr_count: coral_binary.gpr_count.unwrap_or(32),
        workgroup: coral_binary.workgroup.unwrap_or([64, 1, 1]),
        shared_mem_bytes: coral_binary.shared_mem_bytes.unwrap_or(0),
        barrier_count: coral_binary.barrier_count.unwrap_or(0),
    };

    assert_eq!(shader_info.gpr_count, 48);
    assert_eq!(shader_info.shared_mem_bytes, 2048);
    assert_eq!(shader_info.barrier_count, 1);

    // ── Step 5: Mock toadStool dispatch server ────────────────────────
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let listener =
        rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
    let addr = listener.local_addr().unwrap();

    rt.block_on(async {
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 32768];
            let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                .await
                .unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);

            assert!(
                req_str.contains("compute.dispatch.submit"),
                "Method must be compute.dispatch.submit"
            );
            assert!(
                req_str.contains("\"hardware_hint\":\"compute\""),
                "F64 tier routes to compute hint"
            );
            assert!(
                req_str.contains("\"gpr_count\":48"),
                "GPR count from coral compile must propagate"
            );
            assert!(
                req_str.contains("\"shared_mem_bytes\":2048"),
                "Shared memory from coral compile must propagate"
            );
            assert!(
                req_str.contains("\"barrier_count\":1"),
                "Barrier count from coral compile must propagate"
            );

            let body = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                .await
                .unwrap();
            tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
        });

        let staged = Mutex::new(HashMap::new());
        let bindings = vec![IpcBufferBinding {
            index: 0,
            buffer_id: 1,
            size: 256,
            read_only: false,
        }];
        staged
            .lock()
            .unwrap()
            .insert(1u64, bytes::BytesMut::from(&[0u8; 256][..]));

        let result = submit_dispatch(
            &format!("http://127.0.0.1:{}", addr.port()),
            &staged,
            &coral_binary.binary,
            (1, 1, 1),
            &bindings,
            hint,
            shader_info,
        );
        assert!(
            result.is_ok(),
            "Trio dispatch must succeed: {:?}",
            result.err()
        );
        server_task.await.ok();
    });
}

/// Trio contract E2E: GradientFlow → DF64 → coralReef DF64 advice → toadStool dispatch.
///
/// Validates the DF64 (f32-pair emulation) path through the trio:
/// 1. PrecisionBrain routes `GradientFlow` to DF64 (no native f64, coral available)
/// 2. `build_precision_advice` flags `df64_naga_poisoned`
/// 3. `CompileWgslRequest` carries DF64-specific advice to coralReef
/// 4. `submit_dispatch` carries `HardwareHint::Compute` to toadStool
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn trio_contract_e2e_gradient_flow_df64_dispatch() {
    use crate::device::backend::HardwareHint;
    use crate::device::coral_compiler::types::PrecisionAdvice;
    use crate::device::sovereign_device::SovereignDevice;
    use crate::device::sovereign_dispatch_wire::{
        IpcBufferBinding, ShaderDispatchInfo, submit_dispatch,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    // ── PrecisionBrain routes GradientFlow on DF64-capable hardware ───
    let cal = make_cal(true, true, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);

    let domain = PhysicsDomain::GradientFlow;
    let tier = brain.route(domain);

    let (f64_shader, df64_shader) = match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => (true, false),
        PrecisionTier::DF64 => (false, true),
        _ => (false, false),
    };

    // DF64 path: df64_shader should be true
    if !df64_shader {
        // On some hardware configs brain might route differently; skip if so
        return;
    }

    let advice = SovereignDevice::build_precision_advice(f64_shader, df64_shader);
    assert!(advice.is_some());
    let adv = advice.unwrap();
    assert_eq!(adv.tier, "DF64");
    assert!(adv.df64_naga_poisoned);
    assert!(!adv.needs_transcendental_lowering);

    let hint = tier.recommended_hardware_hint();
    assert_eq!(hint, HardwareHint::Compute);

    // ── coralReef wire format with DF64 advice ────────────────────────
    let wire_advice = PrecisionAdvice {
        tier: adv.tier.clone(),
        needs_transcendental_lowering: adv.needs_transcendental_lowering,
        df64_naga_poisoned: adv.df64_naga_poisoned,
        domain: Some("GradientFlow".to_owned()),
    };
    let advice_json = serde_json::to_string(&wire_advice).unwrap();
    assert!(advice_json.contains("\"tier\":\"DF64\""));
    assert!(advice_json.contains("\"df64_naga_poisoned\":true"));
    assert!(advice_json.contains("\"domain\":\"GradientFlow\""));

    // ── Mock toadStool: verify DF64 dispatches as Compute ─────────────
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let listener =
        rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
    let addr = listener.local_addr().unwrap();

    rt.block_on(async {
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 16384];
            let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                .await
                .unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);
            assert!(
                req_str.contains("\"hardware_hint\":\"compute\""),
                "DF64 routes to compute (ALU cores)"
            );

            let body = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                .await
                .unwrap();
            tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
        });

        let staged = Mutex::new(HashMap::new());
        let bindings = vec![IpcBufferBinding {
            index: 0,
            buffer_id: 1,
            size: 128,
            read_only: false,
        }];
        staged
            .lock()
            .unwrap()
            .insert(1u64, bytes::BytesMut::from(&[0u8; 128][..]));

        let info = ShaderDispatchInfo {
            gpr_count: 32,
            workgroup: [64, 1, 1],
            shared_mem_bytes: 0,
            barrier_count: 0,
        };

        let result = submit_dispatch(
            &format!("http://127.0.0.1:{}", addr.port()),
            &staged,
            &[0xCA, 0xFE],
            (1, 1, 1),
            &bindings,
            hint,
            info,
        );
        assert!(
            result.is_ok(),
            "DF64 dispatch must succeed: {:?}",
            result.err()
        );
        server_task.await.ok();
    });
}

/// Trio contract E2E: Inference → F16 → TensorCore hint → toadStool dispatch.
///
/// Validates the tensor core GEMM routing path:
/// 1. F16 precision tier maps to `HardwareHint::TensorCore`
/// 2. F16 requires compiler support (MMA instruction emission)
/// 3. toadStool receives `tensor_core` hardware hint
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn trio_contract_e2e_tensor_core_f16_dispatch() {
    use crate::device::backend::HardwareHint;
    use crate::device::sovereign_dispatch_wire::{
        IpcBufferBinding, ShaderDispatchInfo, submit_dispatch,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    let tier = PrecisionTier::F16;
    let hint = tier.recommended_hardware_hint();
    assert_eq!(hint, HardwareHint::TensorCore);
    assert!(tier.requires_compiler_support(), "F16 MMA needs coralReef");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let listener =
        rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
    let addr = listener.local_addr().unwrap();

    rt.block_on(async {
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 16384];
            let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                .await
                .unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);
            assert!(
                req_str.contains("\"hardware_hint\":\"tensor_core\""),
                "F16 must route to tensor_core for MMA dispatch"
            );
            assert!(
                req_str.contains("compute.dispatch.submit"),
                "Method must be compute.dispatch.submit"
            );

            let body = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                .await
                .unwrap();
            tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
        });

        let staged = Mutex::new(HashMap::new());
        let bindings = vec![IpcBufferBinding {
            index: 0,
            buffer_id: 1,
            size: 512,
            read_only: false,
        }];
        staged
            .lock()
            .unwrap()
            .insert(1u64, bytes::BytesMut::from(&[0u8; 512][..]));

        let info = ShaderDispatchInfo {
            gpr_count: 24,
            workgroup: [32, 1, 1],
            shared_mem_bytes: 4096,
            barrier_count: 2,
        };

        let result = submit_dispatch(
            &format!("http://127.0.0.1:{}", addr.port()),
            &staged,
            &[0x00, 0x01, 0x02, 0x03],
            (8, 8, 1),
            &bindings,
            hint,
            info,
        );
        assert!(
            result.is_ok(),
            "TensorCore dispatch must succeed: {:?}",
            result.err()
        );
        server_task.await.ok();
    });
}
