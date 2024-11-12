#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <vector>

std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec);
std::vector<float> elementwise_add(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> elementwise_mul(const std::vector<float>& a, const std::vector<float>& b);
std::vector<std::vector<float>> transpose_matrix(const std::vector<std::vector<float>>& matrix);
float compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target);
std::vector<float> compute_mse_loss_gradient(const std::vector<float>& output, const std::vector<float>& target);
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> sliding_windows(const std::vector<float>& data, int window_size, int prediction_len);
std::vector<float> matrix_vector_mul_transpose(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec);
std::vector<std::vector<float>> outer_product(const std::vector<float>& a, const std::vector<float>& b);
std::vector<std::vector<float>> matrix_add(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);



#endif // MATRIX_UTILS_HPP
