// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn test_hmc_trajectory_creation() {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let config = GpuHmcConfig {
        nt: 2,
        nx: 2,
        ny: 2,
        nz: 2,
        ..Default::default()
    };
    let hmc = GpuHmcTrajectory::new(device, config).unwrap();
    assert_eq!(hmc.volume, 16);
}
