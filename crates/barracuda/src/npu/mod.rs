// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU Backend Module
//!
//! Event codec and constants for neuromorphic processor integration.
//! Actual hardware execution is handled by the consuming primal or orchestrator.

pub mod constants;
pub mod event_codec;

pub use event_codec::EventCodec;
