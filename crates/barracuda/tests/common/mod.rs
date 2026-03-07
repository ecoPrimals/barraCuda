// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared test utilities for barracuda integration tests.
//!
//! Provides `run_gpu_resilient_async` to wrap GPU test bodies and gracefully
//! skip when NVK/Nouveau driver invalidates resources under concurrent load.
//! Wraps the entire async body in `tokio::time::timeout` to prevent indefinite
//! hangs when the GPU stalls (driver lockup, compute shader deadlock).

use std::panic::{AssertUnwindSafe, catch_unwind};

use barracuda::device::test_pool::GPU_TEST_TIMEOUT;

const NVK_SKIP_PATTERNS: &[&str] = &["does not exist", "device lost", "Parent device"];

fn is_nvk_driver_error(msg: &str) -> bool {
    NVK_SKIP_PATTERNS.iter().any(|p| msg.contains(p))
}

/// Runs a GPU test body, gracefully skipping if the GPU device panics
/// due to NVK driver resource invalidation under concurrent load.
///
/// Wraps the async body in `tokio::time::timeout` (see `GPU_TEST_TIMEOUT`)
/// to prevent indefinite hangs when the GPU stalls. Also acquires a
/// [`GpuTestGate`](barracuda::device::test_harness::GpuTestGate) permit
/// to coordinate concurrent GPU test execution.
///
/// Returns `true` if the test completed successfully, `false` if it should
/// be skipped (NVK driver limitation). Re-panics on other panics (including
/// timeout).
///
/// Spawns a dedicated thread with its own tokio runtime to avoid nesting
/// runtimes when called from `#[tokio::test]`.
pub fn run_gpu_resilient_async<F, Fut>(f: F) -> bool
where
    F: FnOnce() -> Fut + Send + std::panic::UnwindSafe + 'static,
    Fut: std::future::Future<Output = ()>,
{
    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build test runtime");

        let wrapped = async {
            barracuda::device::test_harness::gpu_section(|| async {
                let fut = f();
                assert!(
                    tokio::time::timeout(GPU_TEST_TIMEOUT, fut).await.is_ok(),
                    "GPU test timed out after {GPU_TEST_TIMEOUT:?} \
                     (possible driver stall or shader deadlock)",
                );
            })
            .await;
        };
        catch_unwind(AssertUnwindSafe(|| rt.block_on(wrapped)))
    });

    match handle.join().expect("test thread panicked") {
        Ok(()) => true,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(std::string::String::as_str)
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");

            if is_nvk_driver_error(msg) {
                eprintln!("GPU test skipped: {msg} (NVK driver limitation)");
                false
            } else {
                std::panic::resume_unwind(e);
            }
        }
    }
}
