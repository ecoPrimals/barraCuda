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
