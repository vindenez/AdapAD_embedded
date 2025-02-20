#include "matrix_utils.hpp"
#include <iostream>
#include <sstream>
#include <cmath> 

std::vector<float> compute_mse_loss_gradient(const std::vector<float>& output, const std::vector<float>& target) {
    std::vector<float> gradient(output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        gradient[i] = 2.0f * (output[i] - target[i]) / output.size();
    }
    return gradient;
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
create_sliding_windows(const std::vector<float>& data, int lookback_len, int prediction_len) {
    std::vector<std::vector<float>> x;
    std::vector<float> y;
    
    for (std::size_t i = lookback_len; i < data.size() - prediction_len + 1; ++i) {
        // Create x window (past values)
        std::vector<float> window(data.begin() + (i - lookback_len), 
                                data.begin() + i);
        x.push_back(window);
        
        // Create y window (next value)
        std::vector<float> target(data.begin() + i,
                                data.begin() + i + prediction_len);
        y.push_back(target[0]);  // Only take the first prediction value
    }
    
    return {x, y};
}