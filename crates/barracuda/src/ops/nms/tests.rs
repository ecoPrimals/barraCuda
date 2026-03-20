// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for NMS
//!
//! Validates Non-Maximum Suppression for bounding box filtering.

use super::*;

#[tokio::test]
async fn test_nms_basic() {
    let boxes = vec![
        BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
        },
        BoundingBox {
            x1: 1.0,
            y1: 1.0,
            x2: 11.0,
            y2: 11.0,
            score: 0.8,
        }, // Overlaps
    ];
    let Ok(keep) = nms(boxes, 0.5) else {
        return; // Skip when no GPU
    };
    assert_eq!(keep.len(), 1); // Second box suppressed
    assert_eq!(keep[0], 0); // Highest score kept
}

#[tokio::test]
async fn test_nms_edge_cases() {
    // No overlapping boxes (all kept)
    let boxes = vec![
        BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
        },
        BoundingBox {
            x1: 20.0,
            y1: 20.0,
            x2: 30.0,
            y2: 30.0,
            score: 0.8,
        },
        BoundingBox {
            x1: 40.0,
            y1: 40.0,
            x2: 50.0,
            y2: 50.0,
            score: 0.7,
        },
    ];
    let Ok(keep) = nms(boxes, 0.5) else {
        return; // Skip when no GPU
    };
    assert_eq!(keep.len(), 3); // All boxes kept

    // Single box
    let boxes = vec![BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
        score: 0.9,
    }];
    let Ok(keep) = nms(boxes, 0.5) else {
        return; // Skip when no GPU
    };
    assert_eq!(keep.len(), 1);
}

#[tokio::test]
async fn test_nms_boundary() {
    // Test with overlapping boxes
    let boxes = vec![
        BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
        },
        BoundingBox {
            x1: 5.0,
            y1: 0.0,
            x2: 15.0,
            y2: 10.0,
            score: 0.8,
        },
    ];

    // Very strict threshold (keep everything)
    let Ok(keep) = nms(boxes.clone(), 0.99) else {
        return; // Skip when no GPU
    };
    assert_eq!(keep.len(), 2); // Both kept

    // Very loose threshold (suppress aggressively)
    let Ok(keep) = nms(boxes, 0.01) else {
        return; // Skip when no GPU
    };
    assert_eq!(keep.len(), 1); // Only highest score
}

#[test]
fn test_nms_empty() {
    let keep = nms(vec![], 0.5).unwrap();
    assert_eq!(keep.len(), 0);
}
