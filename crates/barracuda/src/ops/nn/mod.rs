// SPDX-License-Identifier: AGPL-3.0-only

//! Full NCHW neural network operations on GPU.
//!
//! These orchestrators wrap the full-featured WGSL shaders in `ops/nn/`
//! that support stride, padding, dilation, and groups.

pub mod conv2d_gpu;

pub use conv2d_gpu::Conv2dGpu;
