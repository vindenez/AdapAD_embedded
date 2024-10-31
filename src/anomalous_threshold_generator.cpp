#include "anomalous_threshold_generator.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include "config.hpp"

AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len)
    : lookback_len(lookback_len),
      prediction_len(prediction_len),
      generator(lookback_len, lstm_unit, prediction_len, lstm_layer, lookback_len) {
}

void AnomalousThresholdGenerator::train(int num_epochs, float learning_rate, const std::vector<float>& data_to_learn) {
    auto [x, y] = sliding_windows(data_to_learn, lookback_len, prediction_len);
    
    generator.train();
    generator.init_adam_optimizer(learning_rate);

    std::cout << "\nGenerator training for " << num_epochs << " epochs..." << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            generator.zero_grad();
            
            // Reshape input to [1, lookback_len, 1]
            std::vector<std::vector<std::vector<float>>> reshaped_x = {
                std::vector<std::vector<float>>(lookback_len, std::vector<float>(1))
            };
            for (int j = 0; j < lookback_len; j++) {
                reshaped_x[0][j][0] = x[i][j];
            }
            
            auto outputs = generator.forward(reshaped_x);
            float loss = compute_mse_loss(outputs, y[i]);
            generator.backward(y[i], "MSE");
            generator.update_parameters_adam(learning_rate);
            epoch_loss += loss;
        }
        
        epoch_loss /= x.size();

        if (epoch % 100 == 0) {
            std::cout << "Generator Training Epoch " << epoch << "/" << num_epochs 
                     << " Loss: " << std::scientific << epoch_loss << std::endl;
        }
    }
}

float AnomalousThresholdGenerator::update(int epoch_update, float lr_update, const std::vector<float>& past_errors, float recent_error) {
    generator.train();
    generator.init_adam_optimizer(lr_update);
    std::vector<float> loss_history;
    
    // Reshape past_errors to [1, lookback_len, 1]
    std::vector<std::vector<std::vector<float>>> reshaped_input = {
        std::vector<std::vector<float>>(lookback_len, std::vector<float>(1))
    };
    for (int i = 0; i < lookback_len; i++) {
        reshaped_input[0][i][0] = past_errors[i];
    }
    
    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        auto predicted_val = generator.forward(reshaped_input);
        float loss = compute_mse_loss(predicted_val, {recent_error});
        
        generator.zero_grad();
        generator.backward({recent_error}, "MSE");
        generator.update_parameters_adam(lr_update);

        if (loss_history.size() > 0 && loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(loss);
    }
    
    return loss_history.empty() ? 0.0f : loss_history.back();
}

void AnomalousThresholdGenerator::train() {
    is_training = true;
    generator.train();
}

void AnomalousThresholdGenerator::eval() {
    is_training = false;
    generator.eval();
}

float AnomalousThresholdGenerator::generate(const std::vector<float>& prediction_errors, float minimal_threshold) {
    if (prediction_errors.size() != lookback_len) {
        std::cerr << "Error: Invalid prediction_errors size in generate(). Expected: " << lookback_len 
                  << ", Got: " << prediction_errors.size() << std::endl;
        return minimal_threshold;
    }

    if (is_training) {
        eval(); 
    }

    // Calculate mean and std of prediction errors for normalization
    float mean_error = std::accumulate(prediction_errors.begin(), prediction_errors.end(), 0.0f) / prediction_errors.size();
    float variance = 0.0f;
    for (const auto& error : prediction_errors) {
        variance += (error - mean_error) * (error - mean_error);
    }
    variance /= prediction_errors.size();
    float std_error = std::sqrt(variance);

    // Normalize prediction errors
    std::vector<float> normalized_errors;
    normalized_errors.reserve(prediction_errors.size());
    for (const auto& error : prediction_errors) {
        normalized_errors.push_back((error - mean_error) / (std_error + 1e-10f));
    }

    // Reshape normalized_errors to [1, lookback_len, 1]
    std::vector<std::vector<std::vector<float>>> reshaped_input = {
        std::vector<std::vector<float>>(lookback_len, std::vector<float>(1))
    };
    for (int i = 0; i < lookback_len; i++) {
        reshaped_input[0][i][0] = normalized_errors[i];
    }

    auto output = generator.forward(reshaped_input);

    if (output.empty()) {
        std::cerr << "Error: output is empty after generator forward pass in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }

    // Denormalize the output and ensure it's not below minimal_threshold
    float threshold = output[0] * std_error + mean_error;
    threshold = std::max(minimal_threshold, threshold);
    
    return threshold;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
AnomalousThresholdGenerator::sliding_windows(const std::vector<float>& data, int window_size, int prediction_len) {
    std::vector<std::vector<float>> x, y;
    for (size_t i = window_size; i < data.size(); ++i) {
        x.push_back(std::vector<float>(data.begin() + i - window_size, data.begin() + i));
        y.push_back(std::vector<float>(data.begin() + i, std::min(data.begin() + i + prediction_len, data.end())));
    }
    return {x, y};
}

float AnomalousThresholdGenerator::compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float error = output[i] - target[i];
        loss += error * error;
    }
    return loss / output.size();
}

std::vector<float> AnomalousThresholdGenerator::compute_mse_loss_gradient(const std::vector<float>& output, const std::vector<float>& target) {
    std::vector<float> gradient(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[i] = 2.0f * (output[i] - target[i]) / output.size();
    }
    return gradient;
}
