#include "normal_data_predictor.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "matrix_utils.hpp"

NormalDataPredictor::NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len)
    : num_layers(lstm_layer), 
      hidden_size(lstm_unit), 
      lookback_len(lookback_len), 
      prediction_len(prediction_len),
      predictor(lookback_len, hidden_size, prediction_len, num_layers, lookback_len) {
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> NormalDataPredictor::train(int num_epochs, float learning_rate, const std::vector<float>& data_to_learn) {
    auto [x, y] = sliding_windows(data_to_learn, lookback_len, prediction_len);
    
    predictor.train();
    predictor.init_adam_optimizer(learning_rate);

    // Calculate global statistics for normalization
    float mean_global = 0.0f;
    float std_global = 0.0f;
    int total_values = 0;
    
    // Calculate mean
    for (const auto& seq : x) {
        for (const auto& val : seq) {
            mean_global += val;
            total_values++;
        }
    }
    mean_global /= total_values;
    
    // Calculate std
    for (const auto& seq : x) {
        for (const auto& val : seq) {
            std_global += (val - mean_global) * (val - mean_global);
        }
    }
    std_global = std::sqrt(std_global / total_values);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            predictor.zero_grad();
            
            // Use global statistics for normalization
            std::vector<std::vector<std::vector<float>>> reshaped_x = {
                std::vector<std::vector<float>>(lookback_len, std::vector<float>(1))
            };
            for (int j = 0; j < lookback_len; j++) {
                reshaped_x[0][j][0] = (x[i][j] - mean_global) / (std_global + 1e-10f);
            }
            
            // Normalize target using same statistics
            std::vector<float> normalized_y;
            for (const auto& val : y[i]) {
                normalized_y.push_back((val - mean_global) / (std_global + 1e-10f));
            }
            
            auto outputs = predictor.forward(reshaped_x);
            float loss = compute_mse_loss(outputs, normalized_y);
            predictor.backward(normalized_y, "MSE");
            predictor.update_parameters_adam(learning_rate);
            epoch_loss += loss;
        }
        
        epoch_loss /= x.size();
        
        if (epoch % 100 == 0) {
            std::cout << "Training Epoch " << epoch << "/" << num_epochs 
                     << " Loss: " << std::scientific << epoch_loss << std::endl;
        }
    }

    return {x, y};
}

std::vector<float> NormalDataPredictor::predict(const std::vector<float>& input) {
    predictor.eval();
    
    // Calculate statistics for normalization
    float mean_input = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float variance = 0.0f;
    for (const auto& val : input) {
        variance += (val - mean_input) * (val - mean_input);
    }
    variance /= input.size();
    float std_input = std::sqrt(variance);

    // Normalize input
    std::vector<std::vector<std::vector<float>>> reshaped_input = {
        std::vector<std::vector<float>>(lookback_len, std::vector<float>(1))
    };
    for (int i = 0; i < lookback_len; i++) {
        reshaped_input[0][i][0] = (input[i] - mean_input) / (std_input + 1e-10f);
    }
    
    auto predicted_val = predictor.forward(reshaped_input);
    
    // Denormalize output
    predicted_val[0] = predicted_val[0] * std_input + mean_input;
    predicted_val[0] = std::max(0.0f, predicted_val[0]);
    return predicted_val;
}

void NormalDataPredictor::update(int epoch_update, float lr_update, const std::vector<float>& past_observations, const std::vector<float>& recent_observation) {
    predictor.train();
    predictor.init_adam_optimizer(lr_update);
    std::vector<float> loss_history;

    // Reshape past_observations to [1, lookback_len, 1]
    std::vector<std::vector<std::vector<float>>> reshaped_input = {
        std::vector<std::vector<float>>(lookback_len, std::vector<float>(1))
    };
    for (int i = 0; i < lookback_len; i++) {
        reshaped_input[0][i][0] = past_observations[i];
    }

    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        auto predicted_val = predictor.forward(reshaped_input);
        float loss = compute_mse_loss(predicted_val, recent_observation);
        
        predictor.zero_grad();
        predictor.backward(recent_observation, "MSE");
        predictor.update_parameters_adam(lr_update);

        // Early stopping only in update - match Python logic
        if (loss_history.size() > 0 && loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(loss);
    }
}



