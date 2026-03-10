// SPDX-License-Identifier: AGPL-3.0-only
//! E2E tests: Complete workflows (biodiversity, soil moisture, combined).

#![expect(clippy::unwrap_used, reason = "tests")]
use super::*;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::kriging_f64::{KrigingF64, VariogramModel};

mod e2e {
    use super::*;

    #[test]
    fn test_biodiversity_pipeline() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let otu_counts = vec![
            150.0, 89.0, 67.0, 45.0, 32.0, 28.0, 21.0, 15.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 1.0,
        ];
        let shannon = fmr.shannon_entropy(&otu_counts).unwrap();
        let simpson = fmr.simpson_index(&otu_counts).unwrap();
        let richness = otu_counts.len() as f64;
        let evenness = shannon / richness.ln();
        assert!(shannon > 0.0 && shannon < richness.ln() + 0.01);
        assert!(simpson > 0.0 && simpson < 1.0);
        assert!(evenness > 0.0 && evenness <= 1.0);
        println!(
            "✓ Biodiversity pipeline: Shannon H' = {:.4}, Simpson D = {:.4}, Evenness = {:.4}",
            shannon, simpson, evenness
        );
    }

    #[test]
    fn test_soil_moisture_mapping() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let sensors = vec![
            (0.0, 0.0, 0.35),
            (100.0, 0.0, 0.28),
            (0.0, 100.0, 0.32),
            (100.0, 100.0, 0.25),
            (50.0, 50.0, 0.30),
        ];
        let targets: Vec<(f64, f64)> = (0..10)
            .flat_map(|i| (0..10).map(move |j| (i as f64 * 10.0, j as f64 * 10.0)))
            .collect();
        let model = VariogramModel::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 75.0,
        };
        let result = kriging.interpolate(&sensors, &targets, model).unwrap();
        assert_eq!(result.values.len(), 100);
        assert_eq!(result.variances.len(), 100);
        for (i, &v) in result.values.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "VWC at point {} out of range: {}",
                i,
                v
            );
        }
        for (i, &var) in result.variances.iter().enumerate() {
            assert!(var >= 0.0, "Negative variance at point {}: {}", i, var);
        }
        let min_vwc = result.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_vwc = result
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "✓ Soil moisture mapping: Grid 10x10, VWC {:.2}% - {:.2}%",
            min_vwc * 100.0,
            max_vwc * 100.0
        );
    }

    #[test]
    fn test_combined_diversity_spatial() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device.clone()).unwrap();
        let kriging = KrigingF64::new(device).unwrap();
        let sites = vec![
            (0.0, 0.0, fmr.shannon_entropy(&[50.0, 30.0, 20.0]).unwrap()),
            (10.0, 0.0, fmr.shannon_entropy(&[40.0, 40.0, 20.0]).unwrap()),
            (0.0, 10.0, fmr.shannon_entropy(&[60.0, 25.0, 15.0]).unwrap()),
            (
                10.0,
                10.0,
                fmr.shannon_entropy(&[35.0, 35.0, 30.0]).unwrap(),
            ),
            (5.0, 5.0, fmr.shannon_entropy(&[45.0, 35.0, 20.0]).unwrap()),
        ];
        let new_sites = vec![(2.5, 2.5), (7.5, 2.5), (2.5, 7.5), (7.5, 7.5)];
        let model = VariogramModel::Gaussian {
            nugget: 0.0,
            sill: 0.05,
            range: 8.0,
        };
        let result = kriging.interpolate(&sites, &new_sites, model).unwrap();
        for (i, &h) in result.values.iter().enumerate() {
            assert!(
                h > 0.5 && h < 2.0,
                "Interpolated Shannon at site {} out of range: {}",
                i,
                h
            );
        }
        println!(
            "✓ Combined diversity + spatial: {} known, {} interpolated",
            sites.len(),
            new_sites.len()
        );
    }
}
