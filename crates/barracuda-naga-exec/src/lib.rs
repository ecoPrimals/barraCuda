// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU interpreter for naga IR — executes WGSL compute shaders without GPU.
//!
//! Parses WGSL source into a [`naga::Module`], validates it, then interprets
//! the compute entry point on CPU. Storage buffers are plain `Vec<u8>` — no
//! GPU driver, no Vulkan stack, no Mesa dependency.
//!
//! ## Supported IR subset
//!
//! - **Types**: `f32`, `f64`, `u32`, `i32`, `bool`, `vec2/3/4<T>` (all scalar kinds), arrays, structs
//! - **Expressions**: literals, access, `AccessIndex`, binary ops (scalar + vector),
//!   unary ops, math builtins (1/2/3-arg including `Clamp`/`Mix`/`SmoothStep`/`Fma`),
//!   type casts, load, compose, splat, select, swizzle, `arrayLength`
//! - **Statements**: block, store, emit, return (with control flow unwinding),
//!   if/else, loop (with break/continue), switch, atomic ops, barriers
//! - **Globals**: `var<storage>`, `var<uniform>`, `var<workgroup>`,
//!   `@builtin(global_invocation_id)`, `@builtin(local_invocation_id)`,
//!   `@builtin(local_invocation_index)`, `@builtin(num_workgroups)`,
//!   `@builtin(workgroup_id)`, `@builtin(workgroup_size)`
//!
//! ## Not yet supported
//!
//! - Subgroup operations
//! - Texture/sampler operations
#![forbid(unsafe_code)]
#![deny(clippy::pedantic)]

mod error;
mod eval;
mod executor;
mod invocation;
mod sim_buffer;
mod value;
mod vector_ops;
mod workgroup;

pub use error::NagaExecError;
pub use executor::NagaExecutor;
pub use sim_buffer::{SimBuffer, SimBufferUsage};
pub use value::Value;
