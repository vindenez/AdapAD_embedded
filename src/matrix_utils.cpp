#include "matrix_utils.hpp"
#include <iostream>
#include <sstream>

#if defined(__ARM_NEON) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    if (matrix.empty() || vec.empty()) {
        throw std::runtime_error("Error: Empty matrix or vector in matrix_vector_mul. Matrix size: " + 
                                 std::to_string(matrix.size()) + ", Vector size: " + std::to_string(vec.size()));
    }

    if (matrix[0].size() != vec.size()) {
        throw std::runtime_error("Error: Dimension mismatch in matrix_vector_mul. Matrix columns: " + 
                                 std::to_string(matrix[0].size()) + ", Vector size: " + std::to_string(vec.size()) +
                                 ", Matrix rows: " + std::to_string(matrix.size()));
    }

    std::vector<float> result(matrix.size(), 0.0f);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
} */

// Optimized for ARMv7 (Embedded) and ARMv8 (AArch64)
// 
std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    if (matrix.empty() || vec.empty()) {
        throw std::runtime_error("Error: Empty matrix or vector in matrix_vector_mul.");
    }

    if (matrix[0].size() != vec.size()) {
        throw std::runtime_error("Error: Dimension mismatch in matrix_vector_mul.");
    }

    std::vector<float> result(matrix.size(), 0.0f);
    
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized ARM implementation
    for (size_t i = 0; i < matrix.size(); ++i) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (size_t j = 0; j < vec.size(); j += 4) {
            float32x4_t v = vld1q_f32(&vec[j]);
            float32x4_t m = vld1q_f32(&matrix[i][j]);
            sum = vmlaq_f32(sum, v, m);
        }
        float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        result[i] = vget_lane_f32(vpadd_f32(sum2, sum2), 0);

        // Handle remaining elements
        for (size_t j = (vec.size() / 4) * 4; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
#else
    // Fallback implementation
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
#endif

    return result;
}

std::vector<float> elementwise_add(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Dimension mismatch in elementwise addition");
    }

    std::vector<float> result(a.size());

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < a.size(); i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vresult = vaddq_f32(va, vb);
        vst1q_f32(&result[i], vresult);
    }

    // Handle remaining elements
    for (size_t i = (a.size() / 4) * 4; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
#endif

    return result;
}

std::vector<float> elementwise_mul(const std::vector<float>& a, const std::vector<float>& b) {
    
    if (a.size() != b.size()) {
        throw std::runtime_error("Dimension mismatch in elementwise multiplication");
    }

    std::vector<float> result(a.size());
    
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < a.size(); i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vresult = vmulq_f32(va, vb);
        vst1q_f32(&result[i], vresult);
    }

    // Handle remaining elements
    for (size_t i = (a.size() / 4) * 4; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
#endif

    return result;
}

std::vector<std::vector<float>> transpose_matrix(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty()) {
        return {};
    }

    size_t num_rows = matrix.size();
    size_t num_cols = matrix[0].size();

    // Check that all rows have the same number of columns
    for (const auto& row : matrix) {
        if (row.size() != num_cols) {
            throw std::runtime_error("All rows must have the same number of columns to transpose the matrix.");
        }
    }

    std::vector<std::vector<float>> transposed(num_cols, std::vector<float>(num_rows));

    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

float compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float error = output[i] - target[i];
        loss += error * error;
    }
    return loss / output.size();
}

std::vector<float> compute_mse_loss_gradient(const std::vector<float>& output, const std::vector<float>& target) {
    std::vector<float> gradient(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[i] = 2.0f * (output[i] - target[i]) / output.size();
    }
    return gradient;
}

std::vector<float> matrix_vector_mul_transpose(const std::vector<std::vector<float>>& matrix, 
                                             const std::vector<float>& vec) {
    
    if (matrix.empty() || matrix[0].empty() || vec.empty()) {
        throw std::invalid_argument("Empty matrix or vector in matrix_vector_mul_transpose");
    }
    
    if (matrix[0].size() != vec.size()) {
        throw std::invalid_argument(
            "Matrix-vector dimension mismatch in transpose multiplication: " +
            std::to_string(matrix[0].size()) + " != " + std::to_string(vec.size())
        );
    }
    
    std::vector<float> result(matrix.size(), 0.0f);
    
    // Perform matrix-vector multiplication with transpose
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    
    return result;
}

std::vector<std::vector<float>> outer_product(const std::vector<float>& a, 
                                            const std::vector<float>& b) {
    
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Error: Empty vectors in outer_product");
    }

    std::vector<std::vector<float>> result(a.size(), std::vector<float>(b.size(), 0.0f));
    
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized implementation
    for (size_t i = 0; i < a.size(); ++i) {
        float32x4_t va = vdupq_n_f32(a[i]);
        for (size_t j = 0; j < b.size(); j += 4) {
            float32x4_t vb = vld1q_f32(&b[j]);
            float32x4_t vresult = vmulq_f32(va, vb);
            vst1q_f32(&result[i][j], vresult);
        }
        // Handle remaining elements
        for (size_t j = (b.size() / 4) * 4; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
#else
    // Standard implementation
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
#endif

    return result;
}

std::vector<std::vector<float>> matrix_add(const std::vector<std::vector<float>>& a, 
                                         const std::vector<std::vector<float>>& b) {
    if (a.empty() || b.empty() || a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Dimension mismatch in matrix addition");
    }

    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized implementation
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); j += 4) {
            float32x4_t va = vld1q_f32(&a[i][j]);
            float32x4_t vb = vld1q_f32(&b[i][j]);
            float32x4_t vresult = vaddq_f32(va, vb);
            vst1q_f32(&result[i][j], vresult);
        }
        // Handle remaining elements
        for (size_t j = (a[0].size() / 4) * 4; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
#else
    // Standard implementation
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
#endif

    return result;
}

std::vector<float> elementwise_subtract(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Dimension mismatch in elementwise subtraction");
    }

    std::vector<float> result(a.size());

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized implementation
    for (size_t i = 0; i < a.size(); i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vresult = vsubq_f32(va, vb);
        vst1q_f32(&result[i], vresult);
    }

    // Handle remaining elements
    for (size_t i = (a.size() / 4) * 4; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
#else
    // Standard implementation
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
#endif

    return result;
}

// Overload for scalar subtraction from vector
std::vector<float> elementwise_subtract(float scalar, const std::vector<float>& vec) {
    std::vector<float> result(vec.size());

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized implementation
    float32x4_t vscalar = vdupq_n_f32(scalar);
    for (size_t i = 0; i < vec.size(); i += 4) {
        float32x4_t vvec = vld1q_f32(&vec[i]);
        float32x4_t vresult = vsubq_f32(vscalar, vvec);
        vst1q_f32(&result[i], vresult);
    }

    // Handle remaining elements
    for (size_t i = (vec.size() / 4) * 4; i < vec.size(); ++i) {
        result[i] = scalar - vec[i];
    }
#else
    // Standard implementation
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = scalar - vec[i];
    }
#endif

    return result;
}

std::vector<float> elementwise_subtract(const std::vector<float>& vec, float scalar) {
    std::vector<float> result(vec.size());

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized implementation
    float32x4_t vscalar = vdupq_n_f32(scalar);
    for (size_t i = 0; i < vec.size(); i += 4) {
        float32x4_t vvec = vld1q_f32(&vec[i]);
        float32x4_t vresult = vsubq_f32(vvec, vscalar);
        vst1q_f32(&result[i], vresult);
    }

    // Handle remaining elements
    for (size_t i = (vec.size() / 4) * 4; i < vec.size(); ++i) {
        result[i] = vec[i] - scalar;
    }
#else
    // Standard implementation
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] - scalar;
    }
#endif

    return result;
}

std::vector<std::vector<float>> extract_weights(const std::vector<std::vector<float>>& weights, 
                                              size_t start_row, 
                                              size_t num_rows) {
    if (start_row + num_rows > weights.size()) {
        throw std::runtime_error("Invalid range in extract_weights");
    }
    
    std::vector<std::vector<float>> result;
    result.reserve(num_rows);
    
    for (size_t i = 0; i < num_rows; ++i) {
        result.push_back(weights[start_row + i]);
    }
    
    return result;
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector sizes don't match in vector_add");
    }
    
    std::vector<float> result(a.size());
    
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized implementation
    for (size_t i = 0; i < a.size(); i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vresult = vaddq_f32(va, vb);
        vst1q_f32(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (size_t i = (a.size() / 4) * 4; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
#else
    // Standard implementation
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
#endif

    return result;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty()) return {};
    
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<float>> result(cols, std::vector<float>(rows));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

std::string vector_to_string(const std::vector<float>& vec, size_t max_elements) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < std::min(vec.size(), max_elements); ++i) {
        if (i > 0) ss << ", ";
        ss << vec[i];
    }
    if (vec.size() > max_elements) ss << ", ...";
    ss << "]";
    return ss.str();
}

std::vector<float> scale_vector(const std::vector<float>& vec, float scale) {
    std::vector<float> result(vec.size());
    
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t vscale = vdupq_n_f32(scale);
    for (size_t i = 0; i < vec.size(); i += 4) {
        float32x4_t v = vld1q_f32(&vec[i]);
        float32x4_t vresult = vmulq_f32(v, vscale);
        vst1q_f32(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (size_t i = (vec.size() / 4) * 4; i < vec.size(); ++i) {
        result[i] = vec[i] * scale;
    }
#else
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scale;
    }
#endif

    return result;
}

std::vector<float> concatenate_vectors(const std::vector<std::vector<float>>& vectors) {
    // Calculate total size needed
    size_t total_size = 0;
    for (const auto& vec : vectors) {
        total_size += vec.size();
    }
    
    // Pre-allocate result vector
    std::vector<float> result;
    result.reserve(total_size);
    
    // Concatenate all vectors
    for (const auto& vec : vectors) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    
    return result;
}

std::vector<float> dtanh_vector(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized implementation
    float32x4_t vone = vdupq_n_f32(1.0f);
    
    for (size_t i = 0; i < x.size(); i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        // tanh²(x)
        float32x4_t vtanh = vsubq_f32(vone, vmulq_f32(vx, vx));
        vst1q_f32(&result[i], vtanh);
    }
    
    // Handle remaining elements
    for (size_t i = (x.size() / 4) * 4; i < x.size(); ++i) {
        float tanh_x = std::tanh(x[i]);
        result[i] = 1.0f - tanh_x * tanh_x;
    }
#else
    // Standard implementation
    for (size_t i = 0; i < x.size(); ++i) {
        float tanh_x = std::tanh(x[i]);
        result[i] = 1.0f - tanh_x * tanh_x;
    }
#endif

    return result;
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
create_sliding_windows(const std::vector<float>& data, int lookback_len, int prediction_len) {
    std::vector<std::vector<float>> x;
    std::vector<float> y;
    
    for (size_t i = lookback_len; i < data.size() - prediction_len + 1; ++i) {
        // Create x window (past values)
        std::vector<float> window(data.begin() + (i - lookback_len), 
                                data.begin() + i);
        x.push_back(window);
        
        // Create y window (next value)
        std::vector<float> target(data.begin() + i,
                                data.begin() + i + prediction_len);
        y.push_back(target[0]);  // Only take the first prediction value
    }
    
    return {x, y};
}