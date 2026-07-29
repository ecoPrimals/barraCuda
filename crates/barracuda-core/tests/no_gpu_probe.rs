// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for `BARRACUDA_NO_GPU_PROBE` env var behavior.
//!
//! These tests manipulate environment variables and require serialization
//! when run via `cargo test` (shared process). Under nextest (one test per
//! process) the mutex is uncontended.
#![expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
#![expect(
    clippy::await_holding_lock,
    reason = "env-mutex guard held across .await is safe under #[tokio::test] current_thread"
)]

use barracuda_core::BarraCudaPrimal;
use barracuda_core::lifecycle::PrimalLifecycle;

static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_set(key: &str, val: &str) {
    // SAFETY: caller holds ENV_MUTEX, so no concurrent env mutation.
    unsafe { std::env::set_var(key, val) };
}

fn env_remove(key: &str) {
    // SAFETY: caller holds ENV_MUTEX, so no concurrent env mutation.
    unsafe { std::env::remove_var(key) };
}

#[tokio::test]
async fn no_gpu_probe_skips_device_enumeration() {
    let _lock = ENV_MUTEX.lock().unwrap();
    env_set("BARRACUDA_NO_GPU_PROBE", "true");
    let mut primal = BarraCudaPrimal::new();
    primal.start().await.unwrap();
    assert!(primal.compute_device().is_none());
    assert!(primal.device().is_none());
    assert_eq!(
        primal.state(),
        barracuda_core::lifecycle::PrimalState::Running
    );
}

#[tokio::test]
async fn no_gpu_probe_1_truthy() {
    let _lock = ENV_MUTEX.lock().unwrap();
    env_set("BARRACUDA_NO_GPU_PROBE", "1");
    let mut primal = BarraCudaPrimal::new();
    primal.start().await.unwrap();
    assert!(primal.compute_device().is_none());
}

#[tokio::test]
async fn no_gpu_probe_yes_truthy() {
    let _lock = ENV_MUTEX.lock().unwrap();
    env_set("BARRACUDA_NO_GPU_PROBE", "yes");
    let mut primal = BarraCudaPrimal::new();
    primal.start().await.unwrap();
    assert!(primal.compute_device().is_none());
}

#[test]
fn no_gpu_probe_false_does_not_skip() {
    let _lock = ENV_MUTEX.lock().unwrap();
    env_set("BARRACUDA_NO_GPU_PROBE", "false");
    assert!(!BarraCudaPrimal::should_skip_gpu_probe());
}

#[test]
fn no_gpu_probe_unset_does_not_skip() {
    let _lock = ENV_MUTEX.lock().unwrap();
    env_remove("BARRACUDA_NO_GPU_PROBE");
    assert!(!BarraCudaPrimal::should_skip_gpu_probe());
}

#[test]
fn no_gpu_probe_zero_does_not_skip() {
    let _lock = ENV_MUTEX.lock().unwrap();
    env_set("BARRACUDA_NO_GPU_PROBE", "0");
    assert!(!BarraCudaPrimal::should_skip_gpu_probe());
}
