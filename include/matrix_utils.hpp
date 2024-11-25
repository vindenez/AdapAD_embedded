#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <utility>

std::vector<float> matrix_vector_mul(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec);
std::vector<float> elementwise_add(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> elementwise_mul(const std::vector<float>& a, const std::vector<float>& b);
std::vector<std::vector<float>> transpose_matrix(const std::vector<std::vector<float>>& matrix);
float compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target);
std::vector<float> compute_mse_loss_gradient(const std::vector<float>& output, const std::vector<float>& target);
std::vector<float> matrix_vector_mul_transpose(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec);
std::vector<std::vector<float>> outer_product(const std::vector<float>& a, const std::vector<float>& b);
std::vector<std::vector<float>> matrix_add(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
std::vector<float> elementwise_subtract(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> elementwise_subtract(float scalar, const std::vector<float>& vec);
std::vector<float> elementwise_subtract(const std::vector<float>& vec, float scalar);
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix);
std::vector<std::vector<float>> extract_weights(const std::vector<std::vector<float>>& weights, 
                                              std::size_t start_row, 
                                              std::size_t num_rows);
std::vector<float> vector_add(const std::vector<float>& a, 
                             const std::vector<float>& b);
std::string vector_to_string(const std::vector<float>& vec, std::size_t max_elements = 5);
std::vector<float> scale_vector(const std::vector<float>& vec, float scale);
std::vector<float> concatenate_vectors(const std::vector<std::vector<float>>& vectors);
std::vector<float> dtanh_vector(const std::vector<float>& x);
std::pair<std::vector<std::vector<float>>, std::vector<float>>
create_sliding_windows(const std::vector<float>& data, int lookback_len, int prediction_len);

#endif // MATRIX_UTILS_HPP
