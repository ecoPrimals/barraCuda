// SPDX-License-Identifier: AGPL-3.0-only
//! Staging Buffers for Unidirectional Compute Pipelines
//!
//! Provides ring buffer abstractions for streaming data to/from the GPU
//! without blocking round-trips.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  INPUT RING (Host → Device)    OUTPUT RING (Device → Host)         │
//! │  ┌─────────────────────────┐   ┌─────────────────────────┐         │
//! │  │ CPU writes continuously │   │ GPU writes results      │         │
//! │  │ GPU reads and computes  │   │ CPU reads in batches    │         │
//! │  └─────────────────────────┘   └─────────────────────────┘         │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Deep Debt Principles
//!
//! - Zero unsafe code
//! - Fire-and-forget input (no waiting for results during upload)
//! - Batched async output (no blocking on readback)
//! - Capability-based configuration

mod pipeline;
mod ring_buffer;
mod stateful;
mod unidirectional;

pub use pipeline::{PipelineBuilder, Stage, StageLink, StreamingPipeline};
pub use ring_buffer::{
    BufferDirection, GpuRingBuffer, RingBufferConfig, RingBufferStats, WriteHandle,
};
pub use stateful::{KernelDispatch, StatefulConfig, StatefulPipeline};
pub use unidirectional::{
    CompletedWork, PipelineStats, UnidirectionalConfig, UnidirectionalPipeline, WorkHandle,
    WorkUnit,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_config_default() {
        let config = RingBufferConfig::default();
        assert!(config.capacity > 0);
        assert!(config.capacity.is_power_of_two());
    }

    #[test]
    fn test_ring_buffer_config_custom() {
        let config = RingBufferConfig::new(1024 * 1024)
            .with_label("test_buffer")
            .for_input();

        assert_eq!(config.capacity, 1024 * 1024);
        assert_eq!(config.label.as_deref(), Some("test_buffer"));
        assert_eq!(config.direction, BufferDirection::HostToDevice);
    }
}
