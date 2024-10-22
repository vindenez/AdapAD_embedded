#include "matrix_utils.hpp"
#include <iostream>

std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
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
}

std::vector<float> elementwise_add(const std::vector<float>& a, const std::vector<float>& b) {
    // Check for dimension match
    if (a.size() != b.size()) {
        std::cerr << "Error: Dimension mismatch in elementwise addition. Vector A size: " << a.size()
                  << ", Vector B size: " << b.size() << std::endl;
        return {};
    }

    // Perform elementwise addition
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

std::vector<float> elementwise_mul(const std::vector<float>& a, const std::vector<float>& b) {
    // Check for dimension match
    if (a.size() != b.size()) {
        std::cerr << "Error: Dimension mismatch in elementwise multiplication. Vector A size: " << a.size()
                  << ", Vector B size: " << b.size() << std::endl;
        return {};
    }

    // Perform elementwise multiplication
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }

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

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> sliding_windows(const std::vector<float>& data, int window_size, int prediction_len) {
    std::vector<std::vector<float>> x, y;
    for (size_t i = window_size; i < data.size(); ++i) {
        x.push_back(std::vector<float>(data.begin() + i - window_size, data.begin() + i));
        y.push_back(std::vector<float>(data.begin() + i, std::min(data.begin() + i + prediction_len, data.end())));
    }
    return {x, y};
}
