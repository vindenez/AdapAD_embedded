#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <vector>

std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec);
std::vector<float> elementwise_add(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> elementwise_mul(const std::vector<float>& a, const std::vector<float>& b);

#endif // MATRIX_UTILS_HPP
