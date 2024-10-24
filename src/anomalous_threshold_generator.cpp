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

    this->train();
    generator.init_adam_optimizer(learning_rate);

    // Normalize training data
    for (auto& window : x) {
        float mean = std::accumulate(window.begin(), window.end(), 0.0f) / window.size();
        float variance = 0.0f;
        for (const auto& val : window) {
            variance += (val - mean) * (val - mean);
        }
        variance /= window.size();
        float std = std::sqrt(variance);

        // Normalize the window
        for (auto& val : window) {
            val = (val - mean) / (std + 1e-10f);
        }
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < x.size(); ++i) {
            generator.zero_grad();

            auto outputs = generator.forward(x[i]);
            
            float loss = compute_mse_loss(outputs, y[i]);
            epoch_loss += loss;
            
            generator.backward(y[i], "MSE");
            generator.update_parameters_adam(learning_rate);
        }

        epoch_loss /= x.size();
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << ", Loss: " << epoch_loss << std::endl;
    }
}

float AnomalousThresholdGenerator::update(int num_epochs, float lr_update, const std::vector<float>& past_errors, float recent_error) {
    this->train();
    generator.init_adam_optimizer(lr_update);

    // Normalize past errors
    float mean = std::accumulate(past_errors.begin(), past_errors.end(), 0.0f) / past_errors.size();
    float variance = 0.0f;
    for (const auto& error : past_errors) {
        variance += (error - mean) * (error - mean);
    }
    variance /= past_errors.size();
    float std = std::sqrt(variance);

    std::vector<float> normalized_errors;
    normalized_errors.reserve(past_errors.size());
    for (const auto& error : past_errors) {
        normalized_errors.push_back((error - mean) / (std + 1e-10f));
    }

    float normalized_recent_error = (recent_error - mean) / (std + 1e-10f);

    float total_loss = 0.0f;
    std::vector<float> loss_history;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto predicted_val = generator.forward(normalized_errors);
        
        float loss = compute_mse_loss(predicted_val, {normalized_recent_error});
        total_loss += loss;
        loss_history.push_back(loss);
        
        generator.zero_grad();
        generator.backward({normalized_recent_error}, "MSE");
        generator.update_parameters_adam(lr_update);

        if (loss_history.size() >= 3) {
            if (loss_history[loss_history.size() - 1] >= loss_history[loss_history.size() - 2] &&
                loss_history[loss_history.size() - 2] >= loss_history[loss_history.size() - 3]) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                break;
            }
        }
    }

    return total_loss / loss_history.size();
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
    std::cout << "prediction_errors size: " << prediction_errors.size() << std::endl;
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

    auto output = generator.forward(normalized_errors);
    std::cout << "generator output size: " << output.size() << std::endl;

    if (output.empty()) {
        std::cerr << "Error: output is empty after generator forward pass in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }

    // Denormalize the output and ensure it's not below minimal_threshold
    float threshold = output[0] * std_error + mean_error;
    threshold = std::max(minimal_threshold, threshold);
    
    std::cout << "Generated threshold: " << threshold << " (minimal: " << minimal_threshold << ")" << std::endl;

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
