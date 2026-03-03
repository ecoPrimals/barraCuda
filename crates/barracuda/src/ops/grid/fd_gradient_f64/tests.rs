// SPDX-License-Identifier: AGPL-3.0-or-later
//! FD gradient f64 tests

use super::*;
use crate::device::test_pool::get_test_device_if_f64_gpu_available;

#[tokio::test]
async fn test_gradient_1d() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };

    let n = 100;
    let dx = 0.1;
    let grad = Gradient1D::new(device, n, dx).unwrap();

    let input: Vec<f64> = (0..n).map(|i| (i as f64 * dx).powi(2)).collect();
    let result = grad.compute(&input).await.unwrap();

    for i in 1..n - 1 {
        let x = i as f64 * dx;
        let expected = 2.0 * x;
        let error = (result[i] - expected).abs();
        assert!(
            error < 0.02,
            "At i={}, got {}, expected {}, error={}",
            i,
            result[i],
            expected,
            error
        );
    }
}

#[tokio::test]
async fn test_gradient_2d() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };

    let nx = 20;
    let ny = 20;
    let dx = 0.1;
    let dy = 0.1;
    let grad = Gradient2D::new(device, nx, ny, dx, dy).unwrap();

    let mut input = vec![0.0; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            let x = ix as f64 * dx;
            let y = iy as f64 * dy;
            input[ix * ny + iy] = x * x + 2.0 * y;
        }
    }

    let (grad_x, grad_y) = grad.compute(&input).await.unwrap();

    assert_eq!(grad_x.len(), nx * ny);
    assert_eq!(grad_y.len(), nx * ny);

    for ix in 1..nx - 1 {
        for iy in 1..ny - 1 {
            let x = ix as f64 * dx;
            let idx = ix * ny + iy;

            let expected_gx = 2.0 * x;
            let error_gx = (grad_x[idx] - expected_gx).abs();
            assert!(
                error_gx < 0.05,
                "grad_x at ({},{}) = {}, expected {}",
                ix,
                iy,
                grad_x[idx],
                expected_gx
            );

            let error_gy = (grad_y[idx] - 2.0).abs();
            assert!(
                error_gy < 0.01,
                "grad_y at ({},{}) = {}, expected 2",
                ix,
                iy,
                grad_y[idx]
            );
        }
    }
}

#[tokio::test]
async fn test_laplacian_2d() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };

    let nx = 20;
    let ny = 20;
    let dx = 0.1;
    let dy = 0.1;
    let lap = Laplacian2D::new(device, nx, ny, dx, dy).unwrap();

    let mut input = vec![0.0; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            let x = ix as f64 * dx;
            let y = iy as f64 * dy;
            input[ix * ny + iy] = x * x + y * y;
        }
    }

    let result = lap.compute(&input).await.unwrap();
    assert_eq!(result.len(), nx * ny);

    for ix in 2..nx - 2 {
        for iy in 2..ny - 2 {
            let idx = ix * ny + iy;
            let expected = 4.0;
            let error = (result[idx] - expected).abs();
            assert!(
                error < 0.01,
                "Laplacian at ({},{}) = {}, expected {}",
                ix,
                iy,
                result[idx],
                expected
            );
        }
    }
}

#[tokio::test]
async fn test_cylindrical_gradient() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };

    let n_rho = 10;
    let n_z = 10;
    let d_rho = 0.2;
    let d_z = 0.2;
    let z_min = -1.0;

    let grad = CylindricalGradient::new(device, n_rho, n_z, d_rho, d_z, z_min).unwrap();

    let mut input = vec![0.0; n_rho * n_z];
    for i_rho in 0..n_rho {
        for i_z in 0..n_z {
            let rho = (i_rho + 1) as f64 * d_rho;
            let z = z_min + (i_z as f64 + 0.5) * d_z;
            input[i_rho * n_z + i_z] = rho * rho + z;
        }
    }

    let (grad_rho, grad_z) = grad.compute(&input).await.unwrap();

    assert_eq!(grad_rho.len(), n_rho * n_z);
    assert_eq!(grad_z.len(), n_rho * n_z);

    for i_rho in 1..n_rho - 1 {
        for i_z in 1..n_z - 1 {
            let rho = (i_rho + 1) as f64 * d_rho;
            let idx = i_rho * n_z + i_z;

            let expected_rho = 2.0 * rho;
            let error_rho = (grad_rho[idx] - expected_rho).abs();
            assert!(
                error_rho < 0.2,
                "grad_rho at ({},{}) = {}, expected {}",
                i_rho,
                i_z,
                grad_rho[idx],
                expected_rho
            );

            let error_z = (grad_z[idx] - 1.0).abs();
            assert!(
                error_z < 0.01,
                "grad_z at ({},{}) = {}, expected 1",
                i_rho,
                i_z,
                grad_z[idx]
            );
        }
    }
}

#[tokio::test]
async fn test_cylindrical_laplacian() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };

    let n_rho = 10;
    let n_z = 10;
    let d_rho = 0.2;
    let d_z = 0.2;
    let z_min = -1.0;

    let lap = CylindricalLaplacian::new(device, n_rho, n_z, d_rho, d_z, z_min).unwrap();

    let mut input = vec![0.0; n_rho * n_z];
    for i_rho in 0..n_rho {
        for i_z in 0..n_z {
            let z = z_min + (i_z as f64 + 0.5) * d_z;
            input[i_rho * n_z + i_z] = z * z;
        }
    }

    let result = lap.compute(&input).await.unwrap();
    assert_eq!(result.len(), n_rho * n_z);

    for i_rho in 2..n_rho - 2 {
        for i_z in 2..n_z - 2 {
            let idx = i_rho * n_z + i_z;
            let expected = 2.0;
            let error = (result[idx] - expected).abs();
            assert!(
                error < 0.1,
                "Laplacian at ({},{}) = {}, expected {}",
                i_rho,
                i_z,
                result[idx],
                expected
            );
        }
    }
}
