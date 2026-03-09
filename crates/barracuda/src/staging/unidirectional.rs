// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unidirectional Compute Pipeline
//!
//! Fire-and-forget API for streaming data through GPU compute.
//!
//! # Philosophy
//!
//! Traditional GPU pattern (bidirectional):
//! ```text
//! CPU: prepare → [wait] → GPU: compute → [wait] → CPU: collect
//! Total time = prepare + transfer_in + compute + transfer_out + collect
//! ```
//!
//! Unidirectional pattern:
//! ```text
//! CPU: prepare → GPU: compute → CPU: collect (all concurrent)
//! Total time = max(prepare, compute, collect)
//! ```
//!
//! # Design
//!
//! - Input ring buffer: CPU writes continuously, GPU reads when ready
//! - Output ring buffer: GPU writes results, CPU reads in batches
//! - Zero round-trip blocking
//! - Bandwidth-aware throttling for simulation

use super::{GpuRingBuffer, RingBufferConfig, WriteHandle};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Configuration for the unidirectional pipeline
#[derive(Debug, Clone)]
pub struct UnidirectionalConfig {
    /// Fraction of bandwidth allocated to input (0.0 to 1.0)
    /// Default: 0.9 (90% input, 10% output)
    pub input_bandwidth_fraction: f64,

    /// Size of input staging buffer in bytes
    pub input_buffer_size: usize,

    /// Size of output staging buffer in bytes
    pub output_buffer_size: usize,

    /// Target input bandwidth in bytes/sec (for throttling simulation)
    /// None = no throttling (use full bandwidth)
    pub target_input_bandwidth: Option<u64>,

    /// Target output bandwidth in bytes/sec (for throttling simulation)
    pub target_output_bandwidth: Option<u64>,

    /// Enable strict mode: panic on any bidirectional access patterns
    pub strict_mode: bool,
}

const DEFAULT_INPUT_BUFFER_BYTES: usize = 128 * 1024 * 1024;
const DEFAULT_OUTPUT_BUFFER_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_INPUT_BW_FRACTION: f64 = 0.9;
const BYTES_PER_MB: usize = 1024 * 1024;
#[cfg(test)]
const LARGE_INPUT_BUFFER_MB: usize = 256;
#[cfg(test)]
const LARGE_OUTPUT_BUFFER_MB: usize = 32;
#[cfg(test)]
const THROTTLE_TEST_BPS: u64 = 1000;

impl Default for UnidirectionalConfig {
    fn default() -> Self {
        Self {
            input_bandwidth_fraction: DEFAULT_INPUT_BW_FRACTION,
            input_buffer_size: DEFAULT_INPUT_BUFFER_BYTES,
            output_buffer_size: DEFAULT_OUTPUT_BUFFER_BYTES,
            target_input_bandwidth: None,
            target_output_bandwidth: None,
            strict_mode: false,
        }
    }
}

impl UnidirectionalConfig {
    /// Create config with custom buffer sizes
    #[must_use]
    pub fn with_sizes(input_mb: usize, output_mb: usize) -> Self {
        Self {
            input_buffer_size: input_mb * BYTES_PER_MB,
            output_buffer_size: output_mb * BYTES_PER_MB,
            ..Default::default()
        }
    }

    /// Set bandwidth throttling for simulation
    #[must_use]
    pub fn with_throttling(mut self, input_bps: u64, output_bps: u64) -> Self {
        self.target_input_bandwidth = Some(input_bps);
        self.target_output_bandwidth = Some(output_bps);
        self
    }

    /// Enable strict unidirectional enforcement
    #[must_use]
    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }
}

/// A work unit submitted to the pipeline
#[derive(Debug, Clone)]
pub struct WorkUnit {
    /// Unique identifier for tracking
    pub id: u64,
    /// Serialized input data
    pub data: Bytes,
    /// Timestamp when submitted
    pub submitted_at: Instant,
}

/// Handle returned when submitting work
#[derive(Debug, Clone)]
pub struct WorkHandle {
    /// Work unit ID
    pub id: u64,
    /// Write handle from ring buffer
    pub write_handle: WriteHandle,
    /// Submission timestamp
    pub submitted_at: Instant,
}

/// Completed work ready for collection
#[derive(Debug)]
pub struct CompletedWork {
    /// Work unit ID (matches submitted ID)
    pub id: u64,
    /// Result data
    pub data: Bytes,
    /// Time from submission to completion
    pub latency: std::time::Duration,
}

/// Statistics for the unidirectional pipeline
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total work units submitted
    pub submitted: u64,
    /// Total work units completed
    pub completed: u64,
    /// Total input bytes
    pub input_bytes: u64,
    /// Total output bytes
    pub output_bytes: u64,
    /// Average latency (nanoseconds)
    pub avg_latency_ns: u64,
    /// Max latency (nanoseconds)
    pub max_latency_ns: u64,
    /// Input throughput (bytes/sec, rolling average)
    pub input_throughput: f64,
    /// Output throughput (bytes/sec, rolling average)
    pub output_throughput: f64,
    /// Throttle delays (total nanoseconds)
    pub throttle_delay_ns: u64,
}

/// Bandwidth throttler for simulation
struct BandwidthThrottler {
    /// Target bytes per second
    target_bps: u64,
    /// Bytes transferred in current window
    window_bytes: u64,
    /// Window start time
    window_start: Instant,
    /// Window duration (1 second)
    window_duration: std::time::Duration,
}

impl BandwidthThrottler {
    fn new(target_bps: u64) -> Self {
        Self {
            target_bps,
            window_bytes: 0,
            window_start: Instant::now(),
            window_duration: std::time::Duration::from_secs(1),
        }
    }

    /// Check if transfer should be throttled, returns delay if needed
    /// Note: Does NOT record bytes - call `record()` separately after successful transfer
    fn check(&mut self, bytes: u64) -> Option<std::time::Duration> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.window_start);

        // Reset window if expired
        if elapsed >= self.window_duration {
            self.window_bytes = 0;
            self.window_start = now;
            return None;
        }

        // Check if we'd exceed target
        if self.window_bytes + bytes > self.target_bps {
            // Calculate required delay
            let remaining = self.window_duration.saturating_sub(elapsed);
            Some(remaining)
        } else {
            // Don't record here - caller must call record() after transfer
            None
        }
    }

    fn record(&mut self, bytes: u64) {
        self.window_bytes += bytes;
    }
}

/// Unidirectional Compute Pipeline
///
/// Manages streaming data flow to GPU with fire-and-forget semantics.
pub struct UnidirectionalPipeline {
    /// Device reference for shader dispatch.
    device: Arc<WgpuDevice>,
    /// Input ring buffer (Host → Device)
    input_buffer: GpuRingBuffer,
    /// Output ring buffer (Device → Host)
    output_buffer: GpuRingBuffer,
    /// Configuration
    config: UnidirectionalConfig,
    /// Next work ID
    next_id: AtomicU64,
    /// In-flight work queue
    in_flight: VecDeque<(u64, Instant)>,
    /// Statistics
    stats: PipelineStats,
    /// Input throttler (for simulation)
    input_throttler: Option<BandwidthThrottler>,
    /// Output throttler for bandwidth simulation (Phase 5+ parity with `input_throttler`).
    #[expect(
        dead_code,
        reason = "phase 5+: parity with input_throttler for bandwidth simulation"
    )]
    output_throttler: Option<BandwidthThrottler>,
    /// Pipeline start time (for throughput calculation)
    start_time: Instant,
}

impl UnidirectionalPipeline {
    /// Create a new unidirectional pipeline
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation or pipeline build fails (e.g. device
    /// lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>, config: UnidirectionalConfig) -> Result<Self> {
        let input_config = RingBufferConfig::new(config.input_buffer_size)
            .for_input()
            .with_label("UnidirectionalInput");

        let output_config = RingBufferConfig::new(config.output_buffer_size)
            .for_output()
            .with_label("UnidirectionalOutput");

        let input_buffer = GpuRingBuffer::new(device.clone(), input_config)?;
        let output_buffer = GpuRingBuffer::new(device.clone(), output_config)?;

        let input_throttler = config.target_input_bandwidth.map(BandwidthThrottler::new);
        let output_throttler = config.target_output_bandwidth.map(BandwidthThrottler::new);

        tracing::info!(
            "Created unidirectional pipeline: input={}MB, output={}MB, strict={}",
            config.input_buffer_size / (1024 * 1024),
            config.output_buffer_size / (1024 * 1024),
            config.strict_mode
        );

        Ok(Self {
            device,
            input_buffer,
            output_buffer,
            config,
            next_id: AtomicU64::new(1),
            in_flight: VecDeque::new(),
            stats: PipelineStats::default(),
            input_throttler,
            output_throttler,
            start_time: Instant::now(),
        })
    }

    /// Submit work to the pipeline (fire-and-forget)
    ///
    /// Returns immediately. Use `poll_results` to collect completed work.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation or GPU dispatch fails (e.g. device
    /// lost or out of memory).
    pub fn submit(&mut self, data: &[u8]) -> Result<WorkHandle> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();

        // Check throttling
        if let Some(throttler) = &mut self.input_throttler {
            if let Some(delay) = throttler.check(data.len() as u64) {
                if self.config.strict_mode {
                    return Err(BarracudaError::execution_failed(format!(
                        "Strict mode: input bandwidth exceeded, would need {}ms delay",
                        delay.as_millis()
                    )));
                }
                // In non-strict mode, we log but proceed
                self.stats.throttle_delay_ns += delay.as_nanos() as u64;
                tracing::trace!(
                    "Input throttle: {}ms delay for {} bytes",
                    delay.as_millis(),
                    data.len()
                );
            }
            throttler.record(data.len() as u64);
        }

        // Write to input ring buffer
        let write_handle = self.input_buffer.write(data)?;

        // Track in-flight
        self.in_flight.push_back((id, now));

        // Update stats
        self.stats.submitted += 1;
        self.stats.input_bytes += data.len() as u64;
        self.update_throughput();

        Ok(WorkHandle {
            id,
            write_handle,
            submitted_at: now,
        })
    }

    /// Try to submit work (non-blocking)
    ///
    /// Returns None if buffer is full, without blocking.
    pub fn try_submit(&mut self, data: &[u8]) -> Option<WorkHandle> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();

        // Check throttling
        if let Some(throttler) = &mut self.input_throttler {
            if throttler.check(data.len() as u64).is_some() {
                return None; // Would exceed bandwidth
            }
            throttler.record(data.len() as u64);
        }

        // Try write to input ring buffer
        let write_handle = self.input_buffer.try_write(data)?;

        // Track in-flight
        self.in_flight.push_back((id, now));

        // Update stats
        self.stats.submitted += 1;
        self.stats.input_bytes += data.len() as u64;

        Some(WorkHandle {
            id,
            write_handle,
            submitted_at: now,
        })
    }

    /// Advance the input read pointer (call after GPU has consumed data)
    pub fn mark_input_consumed(&mut self, bytes: usize) {
        self.input_buffer.advance_read(bytes);
    }

    /// Mark work as completed (for output tracking)
    ///
    /// Returns the latency for this work unit.
    pub fn mark_completed(&mut self, id: u64, output_size: usize) -> Option<std::time::Duration> {
        // Find and remove from in-flight
        if let Some(pos) = self.in_flight.iter().position(|(wid, _)| *wid == id) {
            let (_, submitted_at) = self.in_flight.remove(pos)?;
            let latency = submitted_at.elapsed();

            // Update stats
            self.stats.completed += 1;
            self.stats.output_bytes += output_size as u64;

            let latency_ns = latency.as_nanos() as u64;
            if self.stats.completed == 1 {
                self.stats.avg_latency_ns = latency_ns;
            } else {
                // Rolling average
                let n = self.stats.completed;
                self.stats.avg_latency_ns = (self.stats.avg_latency_ns * (n - 1) + latency_ns) / n;
            }
            self.stats.max_latency_ns = self.stats.max_latency_ns.max(latency_ns);

            self.update_throughput();

            Some(latency)
        } else {
            tracing::warn!("Unknown work ID {} marked complete", id);
            None
        }
    }

    /// Get current pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get input buffer reference (for compute shader binding)
    pub fn input_gpu_buffer(&self) -> &wgpu::Buffer {
        self.input_buffer.gpu_buffer()
    }

    /// Get output buffer reference (for compute shader binding)
    pub fn output_gpu_buffer(&self) -> &wgpu::Buffer {
        self.output_buffer.gpu_buffer()
    }

    /// Get available input space
    pub fn input_available(&self) -> usize {
        self.input_buffer.available_write()
    }

    /// Get available output data
    pub fn output_available(&self) -> usize {
        self.output_buffer.available_read()
    }

    /// Get count of in-flight work units
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.input_buffer.reset();
        self.output_buffer.reset();
        self.in_flight.clear();
        self.stats = PipelineStats::default();
        self.start_time = Instant::now();
        tracing::info!("Unidirectional pipeline reset");
    }

    /// Get configuration
    pub fn config(&self) -> &UnidirectionalConfig {
        &self.config
    }

    /// The underlying compute device for shader dispatch.
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    fn update_throughput(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.stats.input_throughput = self.stats.input_bytes as f64 / elapsed;
            self.stats.output_throughput = self.stats.output_bytes as f64 / elapsed;
        }
    }
}

impl std::fmt::Debug for UnidirectionalPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnidirectionalPipeline")
            .field("input_buffer", &self.input_buffer)
            .field("output_buffer", &self.output_buffer)
            .field("in_flight", &self.in_flight.len())
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;
    use std::sync::Arc;

    #[test]
    fn test_config_defaults() {
        let config = UnidirectionalConfig::default();
        assert_eq!(config.input_bandwidth_fraction, 0.9);
        assert!(!config.strict_mode);
    }

    #[test]
    fn test_config_with_sizes() {
        let config =
            UnidirectionalConfig::with_sizes(LARGE_INPUT_BUFFER_MB, LARGE_OUTPUT_BUFFER_MB);
        assert_eq!(
            config.input_buffer_size,
            LARGE_INPUT_BUFFER_MB * BYTES_PER_MB
        );
        assert_eq!(
            config.output_buffer_size,
            LARGE_OUTPUT_BUFFER_MB * BYTES_PER_MB
        );
    }

    #[test]
    fn test_config_throttling() {
        let gbps_10 = 10_000_000_000u64; // 10 GB/s
        let gbps_1 = 1_000_000_000u64; // 1 GB/s

        let config = UnidirectionalConfig::default().with_throttling(gbps_10, gbps_1);

        assert_eq!(config.target_input_bandwidth, Some(gbps_10));
        assert_eq!(config.target_output_bandwidth, Some(gbps_1));
    }

    #[test]
    fn test_bandwidth_throttler() {
        let mut throttler = BandwidthThrottler::new(THROTTLE_TEST_BPS);

        // First 500 bytes should pass
        assert!(throttler.check(500).is_none());
        throttler.record(500);

        // Next 400 bytes should pass
        assert!(throttler.check(400).is_none());
        throttler.record(400);

        // Next 200 bytes would exceed, should throttle
        assert!(throttler.check(200).is_some());
    }

    #[tokio::test]
    async fn test_unidirectional_pipeline_new_requires_gpu() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let config = UnidirectionalConfig::with_sizes(4, 2);
        let pipeline = UnidirectionalPipeline::new(Arc::clone(&device), config).unwrap();
        assert_eq!(pipeline.in_flight_count(), 0);
    }

    #[tokio::test]
    async fn test_unidirectional_pipeline_submit_try_submit() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let config = UnidirectionalConfig::with_sizes(4, 2);
        let mut pipeline = UnidirectionalPipeline::new(Arc::clone(&device), config).unwrap();
        let handle = pipeline.submit(b"hello").unwrap();
        assert_eq!(handle.id, 1);
        assert_eq!(handle.write_handle.size, 5);
        let handle2 = pipeline.try_submit(b"world").unwrap();
        assert_eq!(handle2.id, 2);
        assert_eq!(pipeline.in_flight_count(), 2);
    }

    #[tokio::test]
    async fn test_unidirectional_pipeline_mark_completed() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let config = UnidirectionalConfig::with_sizes(4, 2);
        let mut pipeline = UnidirectionalPipeline::new(Arc::clone(&device), config).unwrap();
        let handle = pipeline.submit(b"hello").unwrap();
        let latency = pipeline.mark_completed(handle.id, 5);
        assert!(latency.is_some());
        assert_eq!(pipeline.in_flight_count(), 0);
        assert_eq!(pipeline.stats().completed, 1);
    }

    #[tokio::test]
    async fn test_unidirectional_pipeline_mark_completed_unknown_id_returns_none() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let config = UnidirectionalConfig::with_sizes(4, 2);
        let mut pipeline = UnidirectionalPipeline::new(Arc::clone(&device), config).unwrap();
        let result = pipeline.mark_completed(999, 0);
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_unidirectional_pipeline_mark_input_consumed_reset() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let config = UnidirectionalConfig::with_sizes(4, 2);
        let mut pipeline = UnidirectionalPipeline::new(Arc::clone(&device), config).unwrap();
        pipeline.submit(b"hello").unwrap();
        pipeline.mark_input_consumed(5);
        pipeline.reset();
        assert_eq!(pipeline.in_flight_count(), 0);
    }

    #[tokio::test]
    async fn test_unidirectional_pipeline_stats() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let config = UnidirectionalConfig::with_sizes(4, 2);
        let mut pipeline = UnidirectionalPipeline::new(Arc::clone(&device), config).unwrap();
        pipeline.submit(b"hello").unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.submitted, 1);
        assert_eq!(stats.input_bytes, 5);
    }

    #[tokio::test]
    async fn test_unidirectional_pipeline_debug() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let config = UnidirectionalConfig::with_sizes(4, 2);
        let pipeline = UnidirectionalPipeline::new(Arc::clone(&device), config).unwrap();
        let s = format!("{pipeline:?}");
        assert!(s.contains("UnidirectionalPipeline"));
    }
}
