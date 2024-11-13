#include "normal_data_predictor.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "matrix_utils.hpp"
#include <chrono>

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

    std::cout << "\nTraining on " << x.size() << " sequences for " << num_epochs << " epochs..." << std::endl;
    
    // Calculate global statistics for normalization (like Python)
    float mean_global = 0.0f;
    float std_global = 0.0f;
    int total_values = 0;
    
    // Calculate mean and std like Python implementation
    for (const auto& seq : x) {
        for (const auto& val : seq) {
            mean_global += val;
            total_values++;
        }
    }
    mean_global /= total_values;
    
    for (const auto& seq : x) {
        for (const auto& val : seq) {
            float diff = val - mean_global;
            std_global += diff * diff;
        }
    }
    std_global = std::sqrt(std_global / total_values);

    // Normalize training data
    for (auto& seq : x) {
        for (auto& val : seq) {
            val = (val - mean_global) / (std_global + 1e-10f);
        }
    }
    for (auto& seq : y) {
        for (auto& val : seq) {
            val = (val - mean_global) / (std_global + 1e-10f);
        }
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            predictor.zero_grad();
            
            std::vector<std::vector<std::vector<float>>> reshaped_input(1);
            reshaped_input[0].push_back(x[i]);
            
            auto outputs = predictor.forward(reshaped_input);
            float loss = compute_mse_loss(outputs, y[i]);
            
            predictor.backward(y[i], "MSE");
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

float NormalDataPredictor::predict(const std::vector<float>& observed) {
    predictor.eval();
    
    if (observed.size() < lookback_len) {
        throw std::runtime_error("Not enough observations for prediction");
    }
    
    std::vector<float> input_sequence(observed.end() - lookback_len, observed.end());
    
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].push_back(input_sequence);
    
    auto predicted_val = predictor.forward(reshaped_input);
    
    // Cap the prediction like in Python
    float last_observed = input_sequence.back();
    return std::max(last_observed - 0.2f, 
                   std::min(last_observed + 0.2f, predicted_val[0]));
}

void NormalDataPredictor::update(int epoch_update, float lr_update, 
                                const std::vector<float>& past_observations, 
                                float recent_observation) {
    predictor.train();
    predictor.init_adam_optimizer(lr_update);
    
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].push_back(past_observations);

    // Convert single float to vector for LSTM
    std::vector<float> target_vec = {recent_observation};
    std::vector<float> loss_history;  // Track loss history for early stopping

    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        auto predicted_val = predictor.forward(reshaped_input);
        float loss = compute_mse_loss(predicted_val, target_vec);
        
        predictor.zero_grad();
        predictor.backward(target_vec, "MSE");
        predictor.update_parameters_adam(lr_update);

        // Early stopping like in Python
        if (!loss_history.empty() && loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(loss);
    }
}



