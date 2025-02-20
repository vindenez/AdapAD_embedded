#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <utility>

std::vector<float> compute_mse_loss_gradient(const std::vector<float>& output, const std::vector<float>& target);
std::pair<std::vector<std::vector<float>>, std::vector<float>>
create_sliding_windows(const std::vector<float>& data, int lookback_len, int prediction_len);

#endif // MATRIX_UTILS_HPP
