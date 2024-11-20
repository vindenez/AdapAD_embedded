#include "anomalous_threshold_generator.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "config.hpp"

AnomalousThresholdGenerator::AnomalousThresholdGenerator(
    int lstm_layer, int lstm_unit, int lookback_len, int prediction_len)
    : lookback_len(lookback_len),
      prediction_len(prediction_len) {
    
    generator = std::make_unique<LSTMPredictor>(
        prediction_len,  // num_classes
        lookback_len,    // input_size
        lstm_unit,       // hidden_size
        lstm_layer,      // num_layers
        lookback_len     // seq_length
    );
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
AnomalousThresholdGenerator::create_sliding_windows(const std::vector<float>& data) {
    return ::create_sliding_windows(data, lookback_len, prediction_len);
}

float AnomalousThresholdGenerator::generate(
    const std::vector<float>& prediction_errors,
    float minimal_threshold) {
    
    generator->eval();
    
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].resize(1);
    reshaped_input[0][0] = prediction_errors;
    
    auto output = generator->forward(reshaped_input);
    auto pred = generator->get_final_prediction(output);
    
    // Clamp to minimal_threshold
    return std::max(minimal_threshold, pred[0] * config::threshold_multiplier);
}

void AnomalousThresholdGenerator::update(
    int epoch_update, float lr_update,
    const std::vector<float>& past_errors, float recent_error) {
    
    generator->train();  // Set to training mode
    
    // Single reshape like Python
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].resize(1);
    reshaped_input[0][0] = past_errors;
    
    std::vector<float> target{recent_error};
    
    // Single forward/backward pass
    generator->reset_states();
    auto output = generator->forward(reshaped_input);
    auto pred = generator->get_final_prediction(output);
    
    // Single update step
    generator->train_step(reshaped_input, target, lr_update);

  float AnomalousThresholdGenerator::generate(const std::vector<float>& prediction_errors, float minimal_threshold) {
    if (prediction_errors.size() != lookback_len) {
        std::cerr << "Error: Invalid prediction_errors size in generate(). Expected: " << lookback_len 
                  << ", Got: " << prediction_errors.size() << std::endl;
        return minimal_threshold;
    }

    if (is_training) {
        eval();  
    }

    auto output = generator.forward(prediction_errors);

    if (output.empty()) {
        std::cerr << "Error: output is empty after generator forward pass in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }

    float threshold = std::max(minimal_threshold, output[0]);
    std::cout << "Generated threshold: " << threshold << " (minimal: " << minimal_threshold << ")" << std::endl;

    return threshold;

}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
AnomalousThresholdGenerator::train(int epoch, float lr, const std::vector<float>& data2learn) {
    if (data2learn.size() < lookback_len + prediction_len) {
        throw std::runtime_error("Not enough data for generator training");
    }
    
    auto windows = create_sliding_windows(data2learn);
    generator->train();  // Set to training mode
    
    for (int e = 0; e < epoch; ++e) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < windows.first.size(); ++i) {
            generator->reset_states();
            
            // Reshape input to match PyTorch's format
            std::vector<std::vector<std::vector<float>>> reshaped_input(1);
            reshaped_input[0].resize(1);
            reshaped_input[0][0] = windows.first[i];
            
            std::vector<float> target{windows.second[i]};
            
            auto output = generator->forward(reshaped_input);
            auto pred = generator->get_final_prediction(output);
            
            // Calculate MSE loss
            float diff = pred[0] - target[0];
            epoch_loss += diff * diff;
            
            generator->train_step(reshaped_input, target, lr);
        }
        
        // Report progress
        if ((e + 1) % 100 == 0) {
            float avg_loss = epoch_loss / windows.first.size();
            std::cout << "Generator Epoch " << (e + 1) << "/" << epoch 
                     << ", Average Loss: " << avg_loss << std::endl;
        }
    }
    
    // Return processed windows in the expected format
    std::vector<std::vector<std::vector<float>>> x3d;
    for (const auto& window : windows.first) {
        std::vector<std::vector<std::vector<float>>> input_tensor(1);
        input_tensor[0].resize(1);
        input_tensor[0][0] = window;
        x3d.push_back(input_tensor[0]);
    }
    
    return {x3d, windows.second};
}