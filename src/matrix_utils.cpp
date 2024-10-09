#include "matrix_utils.hpp"
#include <iostream>

std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    // Check if matrix or vector is empty
    if (matrix.empty() || vec.empty()) {
        std::cerr << "Error: Empty matrix or vector in matrix_vector_mul." << std::endl;
        return {};
    }

    // Check for consistent matrix row size
    size_t num_columns = matrix[0].size();
    for (const auto& row : matrix) {
        if (row.size() != num_columns) {
            std::cerr << "Error: Inconsistent row size in matrix." << std::endl;
            return {};
        }
    }

    // Print matrix and vector dimensions for debugging
    std::cout << "Matrix size: " << matrix.size() << " x " << num_columns << ", Vector size: " << vec.size() << std::endl;

    // Check if dimensions match for multiplication
    if (num_columns != vec.size()) {
        std::cerr << "Error: Dimension mismatch in matrix-vector multiplication. Matrix columns: " 
                  << num_columns << ", Vector size: " << vec.size() << std::endl;
        return {};
    }

    // Perform matrix-vector multiplication
    std::vector<float> result(matrix.size(), 0.0f);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < num_columns; ++j) {
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
