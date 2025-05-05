#include "matrix_utils.hpp"
#include <cmath>
#include <iostream>
#include <sstream>

std::pair<std::vector<std::vector<float>>, std::vector<float>>
create_sliding_windows(const std::vector<float> &data, int lookback_len, int prediction_len) {
    std::vector<std::vector<float>> x;
    std::vector<float> y;

    for (std::size_t i = lookback_len; i < data.size() - prediction_len + 1; ++i) {
        // Create x window (past values)
        std::vector<float> window(data.begin() + (i - lookback_len), data.begin() + i);
        x.push_back(window);

        // Create y window (next value)
        std::vector<float> target(data.begin() + i, data.begin() + i + prediction_len);
        y.push_back(target[0]);
    }

    return {x, y};
}
