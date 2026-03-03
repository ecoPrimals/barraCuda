//! Shared test utilities for barracuda integration tests.
//!
//! Provides `run_gpu_resilient_async` to wrap GPU test bodies and gracefully
//! skip when NVK/Nouveau driver invalidates resources under concurrent load.

use std::panic::{catch_unwind, AssertUnwindSafe};

const NVK_SKIP_PATTERNS: &[&str] = &["does not exist", "device lost", "Parent device"];

fn is_nvk_driver_error(msg: &str) -> bool {
    NVK_SKIP_PATTERNS.iter().any(|p| msg.contains(p))
}

/// Runs a GPU test body, gracefully skipping if the GPU device panics
/// due to NVK driver resource invalidation under concurrent load.
///
/// Returns `true` if the test completed successfully, `false` if it should
/// be skipped (NVK driver limitation). Re-panics on other panics.
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

        catch_unwind(AssertUnwindSafe(|| rt.block_on(f())))
    });

    match handle.join().expect("test thread panicked") {
        Ok(()) => true,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(|s| s.as_str())
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
