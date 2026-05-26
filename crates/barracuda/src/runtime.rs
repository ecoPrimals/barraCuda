// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tokio runtime bridging utilities.
//!
//! Provides [`tokio_block_on`] for synchronously running futures from both
//! sync and async contexts without panicking. Used by production code
//! (dispatch config, benchmark, PPPM FFT) and by test helpers.

use std::sync::OnceLock;

/// Block on a future, compatible with both sync and tokio contexts.
///
/// - Multi-threaded runtime: uses `block_in_place` (zero overhead).
/// - Current-thread runtime: spawns an OS thread with a dedicated runtime
///   (avoids both the `block_in_place` panic and nested `block_on` issues).
/// - No runtime: uses a lazily-created static runtime directly.
///
/// # Panics
///
/// Panics if tokio runtime creation fails, or if a spawned thread panics.
pub fn tokio_block_on<F>(f: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

    fn get_or_create_rt() -> &'static tokio::runtime::Runtime {
        RT.get_or_init(|| tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"))
    }

    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let can_block_in_place =
                std::panic::catch_unwind(|| tokio::task::block_in_place(|| {})).is_ok();
            if can_block_in_place {
                tokio::task::block_in_place(|| handle.block_on(f))
            } else {
                std::thread::scope(|s| {
                    s.spawn(|| get_or_create_rt().block_on(f))
                        .join()
                        .expect("tokio_block_on: spawned thread panicked")
                })
            }
        }
        Err(_) => get_or_create_rt().block_on(f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_on_from_sync_context() {
        let result = tokio_block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn block_on_with_async_sleep() {
        let result = tokio_block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            "done"
        });
        assert_eq!(result, "done");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn block_on_from_multi_thread_runtime() {
        let result = tokio_block_on(async { 7 * 6 });
        assert_eq!(result, 42);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn block_on_from_current_thread_runtime() {
        let result = tokio_block_on(async { 2 + 2 });
        assert_eq!(result, 4);
    }

    #[test]
    fn block_on_sequential_calls() {
        let a = tokio_block_on(async { 10 });
        let b = tokio_block_on(async { 20 });
        assert_eq!(a + b, 30);
    }

    #[test]
    fn block_on_spawns_task() {
        let result = tokio_block_on(async {
            let handle = tokio::spawn(async { 99 });
            handle.await.unwrap()
        });
        assert_eq!(result, 99);
    }
}
