#include "matrix_utils.hpp"
#include <cmath>
#include <iostream>
#include <sstream>

std::pair<std::vector<std::vector<float>>, std::vector<float>>
create_sliding_windows(const std::vector<float> &data, int lookback_len, int prediction_len) {
    std::vector<std::vector<float>> x;
    std::vector<float> y;
    
    if (data.size() < lookback_len + prediction_len) {
        return {x, y};
    }
    
    for (size_t i = 0; i <= data.size() - lookback_len - prediction_len; i++) {
        std::vector<float> window;
        for (int j = 0; j < lookback_len; j++) {
            window.push_back(data[i + j]);
        }
        x.push_back(window);
        
        y.push_back(data[i + lookback_len]);
    }
    
    return {x, y};
}
