// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE NTT and pointwise multiplication handlers.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use crate::BarraCudaPrimal;
use serde_json::Value;

/// `barracuda.fhe.ntt` — Execute Number Theoretic Transform on GPU.
///
/// Validates all parameters before checking device availability.
pub(super) async fn fhe_ntt(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(modulus) = params.get("modulus").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: modulus");
    };
    let Some(degree) = params.get("degree").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: degree");
    };
    let Some(root_of_unity) = params.get("root_of_unity").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: root_of_unity");
    };
    let Some(coefficients) = params.get("coefficients").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: coefficients");
    };

    let poly: Vec<u64> = coefficients.iter().filter_map(|v| v.as_u64()).collect();
    let Ok(degree_usize) = usize::try_from(degree) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("degree {degree} too large"));
    };
    let Ok(degree_u32) = u32::try_from(degree) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("degree {degree} exceeds u32::MAX"),
        );
    };
    if poly.len() != degree_usize {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("coefficients length {} != degree {degree}", poly.len()),
        );
    }

    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentional u64→u32 split for FHE coefficient layout"
    )]
    let u32_pairs: Vec<u32> = poly
        .iter()
        .flat_map(|&x| [x as u32, (x >> 32) as u32])
        .collect();
    let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();

    let input_tensor = match barracuda::tensor::Tensor::from_data(
        &f32_bits,
        vec![poly.len() * 2],
        std::sync::Arc::new(dev),
    ) {
        Ok(t) => t,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Tensor creation failed: {e}"),
            );
        }
    };

    let ntt = match barracuda::ops::fhe_ntt::FheNtt::new(
        input_tensor,
        degree_u32,
        modulus,
        root_of_unity,
    ) {
        Ok(n) => n,
        Err(e) => {
            return JsonRpcResponse::error(id, INTERNAL_ERROR, format!("NTT setup failed: {e}"));
        }
    };

    match ntt.execute() {
        Ok(result_tensor) => match result_tensor.to_vec_u32() {
            Ok(u32_data) => {
                let result_u64: Vec<u64> = u32_data
                    .chunks(2)
                    .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
                    .collect();
                JsonRpcResponse::success(
                    id,
                    serde_json::json!({
                        "status": "completed",
                        "modulus": modulus,
                        "degree": degree,
                        "result": result_u64,
                    }),
                )
            }
            Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Readback failed: {e}")),
        },
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("NTT execution failed: {e}")),
    }
}

/// `barracuda.fhe.pointwise_mul` — Execute pointwise polynomial multiplication on GPU.
///
/// Validates all parameters before checking device availability.
pub(super) async fn fhe_pointwise_mul(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(modulus) = params.get("modulus").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: modulus");
    };
    let Some(degree) = params.get("degree").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: degree");
    };
    let Some(a_coeffs) = params.get("a").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: a");
    };
    let Some(b_coeffs) = params.get("b").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: b");
    };

    let a: Vec<u64> = a_coeffs.iter().filter_map(|v| v.as_u64()).collect();
    let b: Vec<u64> = b_coeffs.iter().filter_map(|v| v.as_u64()).collect();

    let Ok(degree_usize) = usize::try_from(degree) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("degree {degree} too large"));
    };
    let Ok(degree_u32) = u32::try_from(degree) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("degree {degree} exceeds u32::MAX"),
        );
    };
    if a.len() != degree_usize || b.len() != degree_usize {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("coefficient arrays must have {degree} elements"),
        );
    }

    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let to_tensor = |poly: &[u64]| -> barracuda::error::Result<barracuda::tensor::Tensor> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "intentional u64→u32 split for FHE coefficient layout"
        )]
        let u32_pairs: Vec<u32> = poly
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();
        barracuda::tensor::Tensor::from_data(
            &f32_bits,
            vec![poly.len() * 2],
            std::sync::Arc::new(dev.clone()),
        )
    };

    let (a_tensor, b_tensor) = match (to_tensor(&a), to_tensor(&b)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Tensor creation failed: {e}"),
            );
        }
    };

    let op = match barracuda::ops::fhe_pointwise_mul::FhePointwiseMul::new(
        a_tensor, b_tensor, degree_u32, modulus,
    ) {
        Ok(op) => op,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Pointwise mul setup failed: {e}"),
            );
        }
    };

    match op.execute() {
        Ok(result_tensor) => match result_tensor.to_vec_u32() {
            Ok(u32_data) => {
                let result_u64: Vec<u64> = u32_data
                    .chunks(2)
                    .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
                    .collect();
                JsonRpcResponse::success(
                    id,
                    serde_json::json!({
                        "status": "completed",
                        "modulus": modulus,
                        "degree": degree,
                        "result": result_u64,
                    }),
                )
            }
            Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Readback failed: {e}")),
        },
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Pointwise mul failed: {e}")),
    }
}
