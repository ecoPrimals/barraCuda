// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::device::SubstrateType;

#[test]
fn test_tiling() {
    let hierarchy = SubstrateMemoryHierarchy {
        substrate_name: "Test GPU".to_string(),
        substrate_type: SubstrateType::Other,
        cache_levels: vec![CacheLevel {
            name: "Cache",
            size_bytes: 64 * 1024 * 1024, // 64 MB
            bandwidth_gbs: 1000.0,
            shared: true,
        }],
        main_memory: MainMemory {
            size_bytes: 16 * 1024 * 1024 * 1024,
            bandwidth_gbs: 500.0,
        },
        optimal_tile_bytes: 32 * 1024 * 1024,
        probed: false,
    };

    let tiler = CacheAwareTiler::new(hierarchy);

    // 1 GB workload
    let config = tiler.optimal_tile_size(
        1024 * 1024 * 1024,
        4,   // f32
        3.0, // A*B+C
    );

    assert!(
        config.tile_bytes < 64 * 1024 * 1024,
        "Tiles should fit in cache"
    );
    assert!(config.num_tiles > 1, "Large workload should be tiled");
}

#[test]
fn test_cache_residency() {
    let hierarchy = SubstrateMemoryHierarchy {
        substrate_name: "Test GPU".to_string(),
        substrate_type: SubstrateType::Other,
        cache_levels: vec![CacheLevel {
            name: "Cache",
            size_bytes: 64 * 1024 * 1024,
            bandwidth_gbs: 1000.0,
            shared: true,
        }],
        main_memory: MainMemory {
            size_bytes: 16 * 1024 * 1024 * 1024,
            bandwidth_gbs: 500.0,
        },
        optimal_tile_bytes: 32 * 1024 * 1024,
        probed: false,
    };

    let tiler = CacheAwareTiler::new(hierarchy);

    // 10 MB should fit
    match tiler.is_cache_resident(10 * 1024 * 1024) {
        CacheResidency::Resident { .. } => {}
        _ => panic!("10 MB should be cache-resident"),
    }

    // 100 MB should overflow
    match tiler.is_cache_resident(100 * 1024 * 1024) {
        CacheResidency::DramBound { .. } => {}
        _ => panic!("100 MB should be DRAM-bound"),
    }
}
