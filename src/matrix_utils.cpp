#include "matrix_utils.hpp"
#include <iostream>

std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    if (matrix.empty() || vec.empty()) {
        throw std::runtime_error("Error: Empty matrix or vector in matrix_vector_mul. Matrix size: " + 
                                 std::to_string(matrix.size()) + ", Vector size: " + std::to_string(vec.size()));
    }

    if (matrix[0].size() != vec.size()) {
        throw std::runtime_error("Error: Dimension mismatch in matrix_vector_mul. Matrix columns: " + 
                                 std::to_string(matrix[0].size()) + ", Vector size: " + std::to_string(vec.size()));
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
