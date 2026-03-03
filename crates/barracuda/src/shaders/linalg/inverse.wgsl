// Matrix Inverse - Gauss-Jordan elimination with augmented matrix
// Computes inverse of square matrices using [A | I] → [I | A^-1]
//
// Algorithm:
// 1. Create augmented matrix [A | I] in work buffer
// 2. Apply Gauss-Jordan elimination with partial pivoting
// 3. Extract inverse from right half of augmented matrix to output
//
// Optimized for small to medium matrices (N <= 32)

struct Params {
    n: u32,  // Matrix size (n x n)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> work_matrix: array<f32>;  // Augmented matrix [A | I]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;      // Final inverse matrix
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    let size = n * n;
    let aug_cols = 2u * n;  // Augmented matrix has 2n columns
    
    // Create augmented matrix [A | I] in work buffer
    // Left half: input matrix A
    // Right half: identity matrix I
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            // Copy input matrix A
            work_matrix[i * aug_cols + j] = input[i * n + j];
            // Initialize identity matrix I
            if (j == i) {
                work_matrix[i * aug_cols + n + j] = 1.0;
            } else {
                work_matrix[i * aug_cols + n + j] = 0.0;
            }
        }
    }
    
    // Gauss-Jordan elimination with partial pivoting
    let epsilon = 1e-6;
    
    for (var i = 0u; i < n; i = i + 1u) {
        // Find pivot (largest absolute value in column i, starting from row i)
        var max_val = abs(work_matrix[i * aug_cols + i]);
        var max_row = i;
        
        for (var k = i + 1u; k < n; k = k + 1u) {
            let val = abs(work_matrix[k * aug_cols + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }
        
        // Check for singularity
        if (max_val < epsilon) {
            // Matrix is singular, output zeros
            for (var j = 0u; j < size; j = j + 1u) {
                output[j] = 0.0;
            }
            return;
        }
        
        // Swap rows if needed (partial pivoting)
        if (max_row != i) {
            for (var j = 0u; j < aug_cols; j = j + 1u) {
                let temp = work_matrix[i * aug_cols + j];
                work_matrix[i * aug_cols + j] = work_matrix[max_row * aug_cols + j];
                work_matrix[max_row * aug_cols + j] = temp;
            }
        }
        
        // Scale pivot row to make pivot = 1
        let pivot = work_matrix[i * aug_cols + i];
        for (var j = 0u; j < aug_cols; j = j + 1u) {
            work_matrix[i * aug_cols + j] = work_matrix[i * aug_cols + j] / pivot;
        }
        
        // Eliminate column i (make all other entries in column i = 0)
        for (var k = 0u; k < n; k = k + 1u) {
            if (k != i) {
                let factor = work_matrix[k * aug_cols + i];
                for (var j = 0u; j < aug_cols; j = j + 1u) {
                    work_matrix[k * aug_cols + j] = work_matrix[k * aug_cols + j] - factor * work_matrix[i * aug_cols + j];
                }
            }
        }
    }
    
    // Extract inverse from right half of augmented matrix
    // Copy right half [I | A^-1] → output [A^-1]
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            output[i * n + j] = work_matrix[i * aug_cols + n + j];
        }
    }
}
