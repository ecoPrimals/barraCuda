// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared JSON-RPC parameter extraction helpers.
//!
//! Used by `math`, `ml`, `spectral`, and other handler modules to parse
//! common parameter shapes (f64 scalars, arrays, 2D matrices) from
//! `serde_json::Value` params without duplication.

use serde_json::Value;

pub(super) fn extract_f64_array(params: &Value, key: &str) -> Option<Vec<f64>> {
    params
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
}

pub(super) fn extract_f64(params: &Value, key: &str) -> Option<f64> {
    params.get(key).and_then(|v| v.as_f64())
}

pub(super) fn extract_matrix(params: &Value, key: &str) -> Option<Vec<Vec<f64>>> {
    params.get(key).and_then(|v| v.as_array()).map(|rows| {
        rows.iter()
            .filter_map(|row| {
                row.as_array()
                    .map(|cols| cols.iter().filter_map(|c| c.as_f64()).collect())
            })
            .collect()
    })
}
