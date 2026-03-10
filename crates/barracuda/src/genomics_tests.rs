// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn test_gc_content() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    // 50% GC
    let seq = b"ATCG";
    assert!((analyzer.gc_content(seq) - 0.5).abs() < 1e-6);

    // 0% GC
    let seq = b"AAAA";
    assert!((analyzer.gc_content(seq) - 0.0).abs() < 1e-6);

    // 100% GC
    let seq = b"GCGC";
    assert!((analyzer.gc_content(seq) - 1.0).abs() < 1e-6);
}

#[test]
fn test_find_pattern() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    let sequence = b"ATCGATCGATCG";
    let pattern = b"TCG";

    let positions = analyzer.find_pattern(sequence, pattern);
    assert_eq!(positions, vec![1, 5, 9]);
}

#[test]
fn test_find_pattern_case_insensitive() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    let sequence = b"atcgATCG";
    let pattern = b"ATCG";

    let positions = analyzer.find_pattern(sequence, pattern);
    assert_eq!(positions, vec![0, 4]);
}

#[test]
fn test_low_complexity_regions() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig {
        complexity_window: 5,
        min_unique_bases: 2,
        parallel_batch: false,
    });

    // AAAAA is low complexity (only 1 unique base)
    let sequence = b"ATCGAAAAAATCG";
    let regions = analyzer.find_low_complexity_regions(sequence);

    // Should find at least one low-complexity region
    assert!(!regions.is_empty(), "Should find low-complexity region");

    // Region should include the repetitive A's
    assert!(
        regions[0].start <= 4,
        "Region should start near or before the A's"
    );
    assert!(
        regions[0].end >= 5,
        "Region should cover at least part of the A's"
    );
}

#[test]
fn test_count_nucleotides() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    let sequence = b"ATCGATCGATCGN";
    let counts = analyzer.count_nucleotides(sequence);

    assert_eq!(counts.a, 3);
    assert_eq!(counts.t, 3);
    assert_eq!(counts.c, 3);
    assert_eq!(counts.g, 3);
    assert_eq!(counts.n, 1);
}

#[test]
fn test_analyze_composition() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    let sequence = b"ATCGATCGATCG";
    let report = analyzer.analyze_composition(sequence).unwrap();

    assert_eq!(report.length, 12);
    assert!((report.gc_content - 0.5).abs() < 1e-6); // 6 G+C out of 12
    assert_eq!(report.nucleotide_counts.a, 3);
}

#[test]
fn test_find_motifs() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    let sequence = b"ATCGATCGATCG";
    let patterns = vec![b"TCG".as_ref(), b"CGA".as_ref()];

    let matches = analyzer.find_motifs(sequence, &patterns).unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].count, 3); // TCG appears 3 times
    assert_eq!(matches[1].count, 2); // CGA appears 2 times
}

#[test]
fn test_quality_filter_pass() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    // Good quality sequence (long, balanced GC, no N's, not low-complexity)
    let sequence = b"ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
    let report = analyzer.quality_filter(sequence).unwrap();

    assert!(report.passes);
    assert_eq!(report.issues.len(), 0);
}

#[test]
fn test_quality_filter_too_short() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig::default());

    let sequence = b"ATCG"; // < 50 bp
    let report = analyzer.quality_filter(sequence).unwrap();

    assert!(!report.passes);
    assert!(report.issues.iter().any(|i| i.contains("too short")));
}

#[test]
fn test_gc_content_batch() {
    let analyzer = SequenceAnalyzer::new(SequenceConfig {
        parallel_batch: false, // Test sequential
        ..Default::default()
    });

    let sequences = vec![b"ATCG".as_ref(), b"GGCC".as_ref(), b"AAAA".as_ref()];
    let gc_values = analyzer.gc_content_batch(&sequences);

    assert_eq!(gc_values.len(), 3);
    assert!((gc_values[0] - 0.5).abs() < 1e-6); // ATCG: 50% GC
    assert!((gc_values[1] - 1.0).abs() < 1e-6); // GGCC: 100% GC
    assert!((gc_values[2] - 0.0).abs() < 1e-6); // AAAA: 0% GC
}
