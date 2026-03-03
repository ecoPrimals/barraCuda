use super::*;

#[test]
fn test_quota_builder() {
    let quota = ResourceQuota::new()
        .with_max_vram_gb(4)
        .with_max_buffers(100)
        .with_preferred_vendor(GpuVendor::Nvidia);
    assert_eq!(quota.max_vram_bytes, Some(4 * 1024 * 1024 * 1024));
    assert_eq!(quota.max_buffers, Some(100));
    assert_eq!(quota.preferred_vendor, Some(GpuVendor::Nvidia));
}

#[test]
fn test_tracker_allocation() {
    let quota = ResourceQuota::new().with_max_vram_mb(100);
    let tracker = QuotaTracker::new(quota);
    assert!(tracker.try_allocate(50 * 1024 * 1024).is_ok());
    assert_eq!(tracker.current_vram_bytes(), 50 * 1024 * 1024);
    assert_eq!(tracker.current_buffers(), 1);
    assert!(tracker.try_allocate(50 * 1024 * 1024).is_ok());
    assert_eq!(tracker.current_vram_bytes(), 100 * 1024 * 1024);
    assert!(tracker.try_allocate(1).is_err());
    assert_eq!(tracker.quota_failures(), 1);
}

#[test]
fn test_tracker_deallocation() {
    let quota = ResourceQuota::new().with_max_vram_mb(100);
    let tracker = QuotaTracker::new(quota);
    tracker.try_allocate(50 * 1024 * 1024).unwrap();
    tracker.try_allocate(50 * 1024 * 1024).unwrap();
    assert!(tracker.try_allocate(1).is_err());
    tracker.deallocate(50 * 1024 * 1024);
    assert!(tracker.try_allocate(1).is_ok());
}

#[test]
fn test_tracker_unlimited() {
    let quota = ResourceQuota::new();
    let tracker = QuotaTracker::new(quota);
    assert!(tracker.try_allocate(1024 * 1024 * 1024).is_ok());
    assert!(tracker.try_allocate(1024 * 1024 * 1024).is_ok());
    assert!(tracker.try_allocate(1024 * 1024 * 1024).is_ok());
}

#[test]
fn test_buffer_count_limit() {
    let quota = ResourceQuota::new().with_max_buffers(2);
    let tracker = QuotaTracker::new(quota);
    assert!(tracker.try_allocate(100).is_ok());
    assert!(tracker.try_allocate(100).is_ok());
    assert!(tracker.try_allocate(100).is_err());
}

#[test]
fn test_single_buffer_limit() {
    let quota = ResourceQuota::new().with_max_single_buffer_bytes(1024);
    let tracker = QuotaTracker::new(quota);
    assert!(tracker.try_allocate(512).is_ok());
    assert!(tracker.try_allocate(1024).is_ok());
    assert!(tracker.try_allocate(1025).is_err());
}

#[test]
fn test_presets() {
    let small = presets::small();
    assert_eq!(small.max_vram_bytes, Some(512 * 1024 * 1024));
    let large = presets::large();
    assert_eq!(large.max_vram_bytes, Some(8 * 1024 * 1024 * 1024));
}

#[test]
fn test_usage_percent() {
    let quota = ResourceQuota::new().with_max_vram_mb(100);
    let tracker = QuotaTracker::new(quota);
    tracker.try_allocate(25 * 1024 * 1024).unwrap();
    assert!((tracker.usage_percent().unwrap() - 25.0).abs() < 0.01);
    tracker.try_allocate(25 * 1024 * 1024).unwrap();
    assert!((tracker.usage_percent().unwrap() - 50.0).abs() < 0.01);
}

#[test]
fn test_summary() {
    let quota = ResourceQuota::named("test").with_max_vram_mb(100);
    let tracker = QuotaTracker::new(quota);
    tracker.try_allocate(50 * 1024 * 1024).unwrap();
    let summary = tracker.summary();
    assert!(summary.contains("test"));
    assert!(summary.contains("50.0%"));
}

#[test]
fn test_device_meets_requirements_min_vram() {
    let quota = ResourceQuota::new().with_min_vram_gb(4);
    assert!(!quota.device_meets_requirements(2 * 1024 * 1024 * 1024, GpuVendor::Nvidia));
    assert!(quota.device_meets_requirements(4 * 1024 * 1024 * 1024, GpuVendor::Nvidia));
    assert!(quota.device_meets_requirements(8 * 1024 * 1024 * 1024, GpuVendor::Nvidia));
}

#[test]
fn test_device_meets_requirements_vendor_preference() {
    let quota = ResourceQuota::new().with_preferred_vendor(GpuVendor::Nvidia);
    assert!(quota.device_meets_requirements(4 * 1024 * 1024 * 1024, GpuVendor::Amd));
    assert!(quota.device_meets_requirements(4 * 1024 * 1024 * 1024, GpuVendor::Nvidia));
}

#[test]
fn test_device_meets_requirements_no_constraints() {
    let quota = ResourceQuota::new();
    assert!(quota.device_meets_requirements(512 * 1024 * 1024, GpuVendor::Intel));
    assert!(quota.device_meets_requirements(1024, GpuVendor::Unknown));
}

#[test]
fn test_would_exceed_quota_total_vram() {
    let quota = ResourceQuota::new().with_max_vram_mb(100);
    let tracker = QuotaTracker::new(quota);
    assert!(!tracker.would_exceed_quota(50 * 1024 * 1024));
    tracker.try_allocate(80 * 1024 * 1024).unwrap();
    assert!(tracker.would_exceed_quota(21 * 1024 * 1024));
    assert!(!tracker.would_exceed_quota(20 * 1024 * 1024));
}

#[test]
fn test_would_exceed_quota_single_buffer() {
    let quota = ResourceQuota::new().with_max_single_buffer_bytes(1024);
    let tracker = QuotaTracker::new(quota);
    assert!(!tracker.would_exceed_quota(1024));
    assert!(tracker.would_exceed_quota(1025));
}

#[test]
fn test_would_exceed_quota_buffer_count() {
    let quota = ResourceQuota::new().with_max_buffers(2);
    let tracker = QuotaTracker::new(quota);
    tracker.try_allocate(100).unwrap();
    tracker.try_allocate(100).unwrap();
    assert!(tracker.would_exceed_quota(100));
}

#[test]
fn test_remaining_vram_bytes() {
    let quota = ResourceQuota::new().with_max_vram_mb(100);
    let tracker = QuotaTracker::new(quota);
    assert_eq!(tracker.remaining_vram_bytes(), Some(100 * 1024 * 1024));
    tracker.try_allocate(30 * 1024 * 1024).unwrap();
    assert_eq!(tracker.remaining_vram_bytes(), Some(70 * 1024 * 1024));
}

#[test]
fn test_remaining_vram_bytes_unlimited() {
    let quota = ResourceQuota::new();
    let tracker = QuotaTracker::new(quota);
    assert_eq!(tracker.remaining_vram_bytes(), None);
}

#[test]
fn test_usage_percent_unlimited() {
    let quota = ResourceQuota::new();
    let tracker = QuotaTracker::new(quota);
    tracker.try_allocate(1024).unwrap();
    assert_eq!(tracker.usage_percent(), None);
}

#[test]
fn test_peak_vram_tracking() {
    let quota = ResourceQuota::new().with_max_vram_mb(100);
    let tracker = QuotaTracker::new(quota);
    tracker.try_allocate(50 * 1024 * 1024).unwrap();
    tracker.try_allocate(30 * 1024 * 1024).unwrap();
    assert_eq!(tracker.peak_vram_bytes(), 80 * 1024 * 1024);
    tracker.deallocate(30 * 1024 * 1024);
    assert_eq!(tracker.current_vram_bytes(), 50 * 1024 * 1024);
    assert_eq!(tracker.peak_vram_bytes(), 80 * 1024 * 1024);
}

#[test]
fn test_total_allocated_bytes() {
    let quota = ResourceQuota::new();
    let tracker = QuotaTracker::new(quota);
    tracker.try_allocate(100).unwrap();
    tracker.try_allocate(200).unwrap();
    tracker.deallocate(100);
    tracker.try_allocate(50).unwrap();
    assert_eq!(tracker.total_allocated_bytes(), 350);
}

#[test]
fn test_new_shared() {
    let quota = ResourceQuota::named("shared_test").with_max_vram_mb(50);
    let tracker = QuotaTracker::new_shared(quota);
    let tracker2 = Arc::clone(&tracker);
    tracker.try_allocate(25 * 1024 * 1024).unwrap();
    assert_eq!(tracker2.current_vram_bytes(), 25 * 1024 * 1024);
}

#[test]
fn test_quota_convenience_methods() {
    let quota_gb = ResourceQuota::new().with_max_vram_gb(2);
    assert_eq!(quota_gb.max_vram_bytes, Some(2 * 1024 * 1024 * 1024));
    let quota_mb = ResourceQuota::new().with_max_vram_mb(512);
    assert_eq!(quota_mb.max_vram_bytes, Some(512 * 1024 * 1024));
    let quota_min_gb = ResourceQuota::new().with_min_vram_gb(1);
    assert_eq!(quota_min_gb.min_vram_bytes, Some(1024 * 1024 * 1024));
}
