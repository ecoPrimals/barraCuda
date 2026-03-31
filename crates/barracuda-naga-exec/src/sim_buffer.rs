// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simulated GPU buffer for CPU-side shader execution.

/// Usage hint for a simulated buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimBufferUsage {
    Storage,
    StorageReadOnly,
    Uniform,
}

/// A simulated GPU buffer — just bytes on the heap.
#[derive(Debug, Clone)]
pub struct SimBuffer {
    pub data: Vec<u8>,
    pub usage: SimBufferUsage,
}

impl SimBuffer {
    /// Create a storage buffer from raw bytes.
    #[must_use]
    pub fn storage(data: Vec<u8>) -> Self {
        Self {
            data,
            usage: SimBufferUsage::Storage,
        }
    }

    /// Create a read-only storage buffer from raw bytes.
    #[must_use]
    pub fn storage_read_only(data: Vec<u8>) -> Self {
        Self {
            data,
            usage: SimBufferUsage::StorageReadOnly,
        }
    }

    /// Create a uniform buffer from raw bytes.
    #[must_use]
    pub fn uniform(data: Vec<u8>) -> Self {
        Self {
            data,
            usage: SimBufferUsage::Uniform,
        }
    }

    /// Create a storage buffer from a slice of f32 values.
    #[must_use]
    pub fn from_f32(values: &[f32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage(data)
    }

    /// Create a read-only storage buffer from a slice of f32 values.
    #[must_use]
    pub fn from_f32_readonly(values: &[f32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage_read_only(data)
    }

    /// Read back as f32 values.
    #[must_use]
    pub fn as_f32(&self) -> Vec<f32> {
        self.data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
            .collect()
    }

    /// Create a storage buffer from a slice of f64 values.
    #[must_use]
    pub fn from_f64(values: &[f64]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage(data)
    }

    /// Read back as f64 values.
    #[must_use]
    pub fn as_f64(&self) -> Vec<f64> {
        self.data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
            .collect()
    }

    /// Create a storage buffer from a slice of u32 values.
    #[must_use]
    pub fn from_u32(values: &[u32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage(data)
    }

    /// Read back as u32 values.
    #[must_use]
    pub fn as_u32(&self) -> Vec<u32> {
        self.data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
            .collect()
    }
}
