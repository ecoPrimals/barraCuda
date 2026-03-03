// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dispatch semaphore — gates concurrent dispatch volume.
//!
//! The device knows its own concurrency budget: CPU/software backends
//! get 2 permits (llvmpipe is effectively single-threaded), discrete
//! GPUs get 8, integrated GPUs get 4. This prevents driver overload
//! without requiring callers to manage thread counts.

const CONCURRENCY_BUDGET_CPU: usize = 2;
const CONCURRENCY_BUDGET_IGPU: usize = 4;
const CONCURRENCY_BUDGET_DGPU: usize = 8;
const CONCURRENCY_BUDGET_DEFAULT: usize = 4;

/// Sync counting semaphore — gates concurrent dispatch volume.
#[derive(Debug)]
pub(crate) struct DispatchSemaphore {
    state: std::sync::Mutex<usize>,
    available: std::sync::Condvar,
    max_permits: usize,
}

impl DispatchSemaphore {
    pub(crate) fn new(max_permits: usize) -> Self {
        Self {
            state: std::sync::Mutex::new(max_permits),
            available: std::sync::Condvar::new(),
            max_permits,
        }
    }

    pub(crate) fn acquire(&self) -> DispatchPermit<'_> {
        let mut count = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while *count == 0 {
            count = self
                .available
                .wait(count)
                .unwrap_or_else(|e| e.into_inner());
        }
        *count -= 1;
        DispatchPermit(self)
    }

    pub(crate) fn max_permits(&self) -> usize {
        self.max_permits
    }
}

impl Clone for DispatchSemaphore {
    fn clone(&self) -> Self {
        Self::new(self.max_permits)
    }
}

/// RAII guard — holds one dispatch permit. Released on drop.
pub struct DispatchPermit<'a>(&'a DispatchSemaphore);

impl Drop for DispatchPermit<'_> {
    fn drop(&mut self) {
        let mut count = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        *count += 1;
        self.0.available.notify_one();
    }
}

/// Determine concurrency budget from adapter type.
pub(crate) fn concurrency_budget(device_type: wgpu::DeviceType) -> usize {
    match device_type {
        wgpu::DeviceType::Cpu => CONCURRENCY_BUDGET_CPU,
        wgpu::DeviceType::IntegratedGpu => CONCURRENCY_BUDGET_IGPU,
        wgpu::DeviceType::DiscreteGpu => CONCURRENCY_BUDGET_DGPU,
        wgpu::DeviceType::VirtualGpu => CONCURRENCY_BUDGET_DEFAULT,
        _ => CONCURRENCY_BUDGET_DEFAULT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrency_budget_returns_positive() {
        assert!(concurrency_budget(wgpu::DeviceType::Cpu) > 0);
        assert!(concurrency_budget(wgpu::DeviceType::IntegratedGpu) > 0);
        assert!(concurrency_budget(wgpu::DeviceType::DiscreteGpu) > 0);
        assert!(concurrency_budget(wgpu::DeviceType::VirtualGpu) > 0);
        assert!(concurrency_budget(wgpu::DeviceType::Other) > 0);
    }

    #[test]
    fn test_dispatch_semaphore_acquire_release() {
        let sem = DispatchSemaphore::new(2);
        assert_eq!(sem.max_permits(), 2);
        let permit = sem.acquire();
        assert_eq!(sem.max_permits(), 2);
        drop(permit);
        let _permit2 = sem.acquire();
    }

    #[test]
    fn test_concurrency_budget_bounded() {
        const MAX_REASONABLE: usize = 256;
        assert!(concurrency_budget(wgpu::DeviceType::Cpu) <= MAX_REASONABLE);
        assert!(concurrency_budget(wgpu::DeviceType::IntegratedGpu) <= MAX_REASONABLE);
        assert!(concurrency_budget(wgpu::DeviceType::DiscreteGpu) <= MAX_REASONABLE);
        assert!(concurrency_budget(wgpu::DeviceType::VirtualGpu) <= MAX_REASONABLE);
        assert!(concurrency_budget(wgpu::DeviceType::Other) <= MAX_REASONABLE);
    }
}
