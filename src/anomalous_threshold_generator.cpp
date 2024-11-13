#include "anomalous_threshold_generator.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include "config.hpp"
#include <chrono>

AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len)
    : lookback_len(lookback_len),
      prediction_len(prediction_len),
      generator(prediction_len,    // num_classes
               lookback_len,      // input_size
               lstm_unit,         // hidden_size
               lstm_layer,        // num_layers
               lookback_len) {    // lookback_len
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
            
            std::vector<std::vector<std::vector<float>>> reshaped_input(1);
            reshaped_input[0] = std::vector<std::vector<float>>(1, x[i]);
            
            auto outputs = generator.forward(reshaped_input);
            std::vector<float> target = y[i];
            float loss = compute_mse_loss(outputs, target);
            
            generator.backward(target, "MSE");
            generator.update_parameters_adam(learning_rate);
            
            epoch_loss += loss;
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Generator Training Epoch " << epoch << "/" << num_epochs 
                     << " Loss: " << std::scientific << epoch_loss/x.size() << std::endl;
        }
    }
    generator.eval();
}

float AnomalousThresholdGenerator::update(int epoch_update, float lr_update, 
                                        const std::vector<float>& past_errors, 
                                        float recent_error) {
    // First get initial prediction in eval mode
    generator.eval();
    std::vector<std::vector<std::vector<float>>> eval_input(1);
    eval_input[0] = std::vector<std::vector<float>>(1, past_errors);
    auto initial_threshold = generator.forward(eval_input);
    
    // Then switch to training mode
    generator.train();
    generator.init_adam_optimizer(lr_update);
    
    // Prepare input in same shape as Python
    std::vector<std::vector<std::vector<float>>> train_input(1);
    train_input[0] = std::vector<std::vector<float>>(1, past_errors);
    
    // Target should be single value in same shape as Python
    std::vector<float> target = {recent_error};
    
    float final_loss = 0.0f;
    std::vector<float> loss_history;  // Track loss history for early stopping
    
    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        generator.zero_grad();
        
        auto predicted_val = generator.forward(train_input);
        float loss = compute_mse_loss(predicted_val, target);
        
        generator.backward(target, "MSE");
        generator.update_parameters_adam(lr_update);
        
        final_loss = loss;
        
        // Early stopping like Python
        if (!loss_history.empty() && loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(loss);
    }
    
    generator.eval();
    return final_loss;
}

void AnomalousThresholdGenerator::train() {
    is_training = true;
    generator.train();
}

void AnomalousThresholdGenerator::eval() {
    is_training = false;
    generator.eval();
}

float AnomalousThresholdGenerator::generate(const std::vector<float>& prediction_errors, 
                                          float minimal_threshold) {
    generator.eval();
    
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0] = std::vector<std::vector<float>>(1, prediction_errors);
    
    auto threshold = generator.forward(reshaped_input);
    float scaled_threshold = threshold[0];
    
    // Ensure threshold is at least minimal_threshold
    scaled_threshold = std::max(scaled_threshold, minimal_threshold);
    
    std::cout << "Debug: Generated threshold=" << scaled_threshold 
              << " (minimal=" << minimal_threshold << ")" << std::endl;
              
    return scaled_threshold;
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
