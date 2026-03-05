// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU Ring Buffer for Streaming Data
//!
//! A lock-free ring buffer for staging data between CPU and GPU
//! in a unidirectional compute pipeline.
//!
//! # Design
//!
//! - Single producer, single consumer (SPSC) pattern
//! - Atomic head/tail pointers for thread safety
//! - Power-of-two capacity for efficient modulo via bitmask
//! - Zero unsafe code

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Direction of data flow for the ring buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferDirection {
    /// CPU → GPU (input staging)
    HostToDevice,
    /// GPU → CPU (output staging)
    DeviceToHost,
}

/// Configuration for creating a ring buffer
#[derive(Debug, Clone)]
pub struct RingBufferConfig {
    /// Buffer capacity in bytes (will be rounded up to power of 2)
    pub capacity: usize,
    /// Data flow direction
    pub direction: BufferDirection,
    /// Optional label for debugging
    pub label: Option<String>,
}

const DEFAULT_RING_BUFFER_BYTES: usize = 64 * 1024 * 1024;

impl Default for RingBufferConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_RING_BUFFER_BYTES,
            direction: BufferDirection::HostToDevice,
            label: None,
        }
    }
}

impl RingBufferConfig {
    /// Create a new config with specified capacity
    pub fn new(capacity: usize) -> Self {
        // Round up to power of 2 for efficient modulo
        let capacity = capacity.next_power_of_two();
        Self {
            capacity,
            ..Default::default()
        }
    }

    /// Set the buffer direction
    pub fn with_direction(mut self, direction: BufferDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Configure for input (Host → Device)
    pub fn for_input(mut self) -> Self {
        self.direction = BufferDirection::HostToDevice;
        self
    }

    /// Configure for output (Device → Host)
    pub fn for_output(mut self) -> Self {
        self.direction = BufferDirection::DeviceToHost;
        self
    }

    /// Set label for debugging
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Statistics for the ring buffer
#[derive(Debug, Clone, Default)]
pub struct RingBufferStats {
    /// Total bytes written
    pub bytes_written: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Number of write operations
    pub write_count: u64,
    /// Number of read operations
    pub read_count: u64,
    /// Number of times write blocked (buffer full)
    pub write_blocks: u64,
    /// Number of times read found no data
    pub read_empty: u64,
}

/// Handle returned from a write operation for tracking
#[derive(Debug, Clone)]
pub struct WriteHandle {
    /// Offset in the ring buffer where data was written
    pub offset: u64,
    /// Size of the written data
    pub size: usize,
    /// Sequence number for ordering
    pub sequence: u64,
}

/// GPU Ring Buffer for streaming data
///
/// Thread-safe SPSC ring buffer backed by GPU memory.
pub struct GpuRingBuffer {
    /// The underlying GPU buffer
    buffer: wgpu::Buffer,
    /// Staging buffer for CPU access (only for DeviceToHost)
    staging_buffer: Option<wgpu::Buffer>,
    /// Buffer capacity (power of 2)
    capacity: usize,
    /// Bitmask for efficient modulo (capacity - 1)
    mask: usize,
    /// Direction of data flow
    direction: BufferDirection,
    /// Write head (where next write goes)
    write_head: AtomicU64,
    /// Read head (where next read comes from)
    read_head: AtomicU64,
    /// Sequence counter for write handles
    sequence: AtomicU64,
    /// Statistics
    stats: RingBufferStats,
    /// Device reference for queue operations
    device: Arc<WgpuDevice>,
    /// Label for debugging
    label: String,
}

impl GpuRingBuffer {
    /// Create a new ring buffer
    pub fn new(device: Arc<WgpuDevice>, config: RingBufferConfig) -> Result<Self> {
        let capacity = config.capacity;
        let mask = capacity - 1;
        let label = config
            .label
            .unwrap_or_else(|| format!("RingBuffer:{:?}", config.direction));

        // Determine buffer usage based on direction
        let usage = match config.direction {
            BufferDirection::HostToDevice => {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST // CPU can write
            }
            BufferDirection::DeviceToHost => {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC // Can copy to staging
            }
        };

        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&label),
            size: capacity as u64,
            usage,
            mapped_at_creation: false,
        });

        // Create staging buffer for output direction
        let staging_buffer = if config.direction == BufferDirection::DeviceToHost {
            Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}:Staging")),
                size: capacity as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        tracing::info!(
            "Created ring buffer '{}': {} bytes, direction={:?}",
            label,
            capacity,
            config.direction
        );

        Ok(Self {
            buffer,
            staging_buffer,
            capacity,
            mask,
            direction: config.direction,
            write_head: AtomicU64::new(0),
            read_head: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            stats: RingBufferStats::default(),
            device,
            label,
        })
    }

    /// Get buffer capacity in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current write head position
    pub fn write_position(&self) -> u64 {
        self.write_head.load(Ordering::Acquire)
    }

    /// Get current read head position
    pub fn read_position(&self) -> u64 {
        self.read_head.load(Ordering::Acquire)
    }

    /// Get available space for writing
    pub fn available_write(&self) -> usize {
        let write = self.write_head.load(Ordering::Acquire);
        let read = self.read_head.load(Ordering::Acquire);

        // Available = capacity - (write - read) - 1 (leave one byte to distinguish full from empty)
        let used = (write.wrapping_sub(read)) as usize;
        if used >= self.capacity {
            0
        } else {
            self.capacity - used - 1
        }
    }

    /// Get available data for reading
    pub fn available_read(&self) -> usize {
        let write = self.write_head.load(Ordering::Acquire);
        let read = self.read_head.load(Ordering::Acquire);

        (write.wrapping_sub(read)) as usize
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.available_read() == 0
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.available_write() == 0
    }

    /// Write data to the ring buffer (for HostToDevice)
    ///
    /// Returns a handle for tracking, or None if not enough space.
    /// This is non-blocking - returns immediately if buffer is full.
    pub fn try_write(&mut self, data: &[u8]) -> Option<WriteHandle> {
        if self.direction != BufferDirection::HostToDevice {
            tracing::warn!("Attempted write to DeviceToHost ring buffer");
            return None;
        }

        if data.len() > self.available_write() {
            self.stats.write_blocks += 1;
            return None;
        }

        let write_pos = self.write_head.load(Ordering::Acquire);
        let offset = (write_pos as usize) & self.mask;
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);

        // Handle wraparound
        let first_part_len = (self.capacity - offset).min(data.len());
        let second_part_len = data.len() - first_part_len;

        // Write first part
        self.device
            .queue
            .write_buffer(&self.buffer, offset as u64, &data[..first_part_len]);

        // Write second part if wrapping
        if second_part_len > 0 {
            self.device
                .queue
                .write_buffer(&self.buffer, 0, &data[first_part_len..]);
        }

        // Update write head
        let new_pos = write_pos.wrapping_add(data.len() as u64);
        self.write_head.store(new_pos, Ordering::Release);

        // Update stats
        self.stats.bytes_written += data.len() as u64;
        self.stats.write_count += 1;

        Some(WriteHandle {
            offset: write_pos,
            size: data.len(),
            sequence: seq,
        })
    }

    /// Write data, blocking until space is available
    ///
    /// For async contexts, prefer `try_write` with polling.
    pub fn write(&mut self, data: &[u8]) -> Result<WriteHandle> {
        // In strict unidirectional mode, we shouldn't block
        // This is a simple spin-wait for now; could be evolved to async
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 1_000_000;

        while self.available_write() < data.len() {
            attempts += 1;
            if attempts > MAX_ATTEMPTS {
                return Err(BarracudaError::execution_failed(format!(
                    "Ring buffer '{}' full after {} attempts ({} bytes needed, {} available)",
                    self.label,
                    attempts,
                    data.len(),
                    self.available_write()
                )));
            }
            std::hint::spin_loop();
        }

        self.try_write(data)
            .ok_or_else(|| BarracudaError::execution_failed("Unexpected write failure"))
    }

    /// Advance the read head (mark data as consumed)
    ///
    /// Call this after processing data to free up space.
    pub fn advance_read(&mut self, bytes: usize) {
        let current = self.read_head.load(Ordering::Acquire);
        let new_pos = current.wrapping_add(bytes as u64);
        self.read_head.store(new_pos, Ordering::Release);

        self.stats.bytes_read += bytes as u64;
        self.stats.read_count += 1;
    }

    /// Get statistics
    pub fn stats(&self) -> &RingBufferStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = RingBufferStats::default();
    }

    /// Get the underlying GPU buffer (for use in compute shaders)
    pub fn gpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the staging buffer for async CPU readback (DeviceToHost only).
    pub fn staging_buffer(&self) -> Option<&wgpu::Buffer> {
        self.staging_buffer.as_ref()
    }

    /// Get buffer direction
    pub fn direction(&self) -> BufferDirection {
        self.direction
    }

    /// Get label
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Reset the ring buffer (clear all data)
    pub fn reset(&mut self) {
        self.write_head.store(0, Ordering::Release);
        self.read_head.store(0, Ordering::Release);
        self.sequence.store(0, Ordering::Release);
        tracing::debug!("Ring buffer '{}' reset", self.label);
    }
}

impl std::fmt::Debug for GpuRingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRingBuffer")
            .field("label", &self.label)
            .field("capacity", &self.capacity)
            .field("direction", &self.direction)
            .field("write_head", &self.write_head.load(Ordering::Relaxed))
            .field("read_head", &self.read_head.load(Ordering::Relaxed))
            .field("available_write", &self.available_write())
            .field("available_read", &self.available_read())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_power_of_two() {
        // Non-power-of-two should be rounded up
        let config = RingBufferConfig::new(1000);
        assert_eq!(config.capacity, 1024);

        let config = RingBufferConfig::new(1025);
        assert_eq!(config.capacity, 2048);

        // Power of two stays the same
        let config = RingBufferConfig::new(4096);
        assert_eq!(config.capacity, 4096);
    }

    #[test]
    fn test_config_directions() {
        let input = RingBufferConfig::new(1024).for_input();
        assert_eq!(input.direction, BufferDirection::HostToDevice);

        let output = RingBufferConfig::new(1024).for_output();
        assert_eq!(output.direction, BufferDirection::DeviceToHost);
    }

    #[test]
    fn test_available_space_calculation() {
        // Mock the atomic calculations
        let capacity = 1024usize;
        let mask = capacity - 1;

        // Empty buffer
        let write: u64 = 0;
        let read: u64 = 0;
        let used = (write.wrapping_sub(read)) as usize;
        let available = if used >= capacity {
            0
        } else {
            capacity - used - 1
        };
        assert_eq!(available, 1023); // capacity - 1

        // Half full
        let write: u64 = 512;
        let read: u64 = 0;
        let used = (write.wrapping_sub(read)) as usize;
        let available = if used >= capacity {
            0
        } else {
            capacity - used - 1
        };
        assert_eq!(available, 511);

        // Test wraparound
        let write: u64 = u64::MAX;
        let read: u64 = u64::MAX - 100;
        let used = (write.wrapping_sub(read)) as usize;
        assert_eq!(used, 100);

        let _ = mask; // Suppress unused warning
    }
}
