#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

std::pair<std::vector<std::vector<float>>, std::vector<float>>
create_sliding_windows(const std::vector<float> &data, int lookback_len, int prediction_len);

#endif // MATRIX_UTILS_HPP
