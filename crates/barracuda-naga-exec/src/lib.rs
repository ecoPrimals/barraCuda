// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU interpreter for naga IR — executes WGSL compute shaders without GPU.
//!
//! Parses WGSL source into a [`naga::Module`], validates it, then interprets
//! the compute entry point on CPU. Storage buffers are plain `Vec<u8>` — no
//! GPU driver, no Vulkan stack, no Mesa dependency.
//!
//! ## Supported IR subset (Phase 2a — elementwise ops)
//!
//! - **Types**: `f32`, `f64`, `u32`, `i32`, `bool`, `vec2/3/4<T>`, arrays, structs
//! - **Expressions**: literals, access, binary ops, unary ops, math builtins,
//!   type casts, load, compose, splat
//! - **Statements**: block, store, emit, return
//! - **Globals**: `var<storage>`, `var<uniform>`, `@builtin(global_invocation_id)`
//!
//! ## Not yet supported (Phase 2b)
//!
//! - `var<workgroup>` shared memory
//! - `workgroupBarrier()`
//! - Atomic operations
//! - Subgroup operations
#![forbid(unsafe_code)]
#![deny(clippy::pedantic)]

mod error;
mod eval;
mod executor;
mod sim_buffer;
mod value;
mod workgroup;

pub use error::NagaExecError;
pub use executor::NagaExecutor;
pub use sim_buffer::{SimBuffer, SimBufferUsage};
pub use value::Value;
