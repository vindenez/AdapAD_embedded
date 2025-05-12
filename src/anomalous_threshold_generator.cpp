#include "anomalous_threshold_generator.hpp"
#include "config.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lstm_layer, int lstm_unit,
                                                         int lookback_len, int prediction_len)
    : lookback_len(lookback_len), prediction_len(prediction_len) {

    generator = LSTMPredictorFactory::create_predictor(prediction_len, // num_classes
                                                       lookback_len,   // input_size
                                                       lstm_unit,      // hidden_size
                                                       lstm_layer,     // num_layers
                                                       lookback_len,   // lookback_len
                                                       true);          // batch_first

    // Initialize layer cache
    generator->initialize_layer_cache();

    // Pre-allocate vectors using the new method
    generator->pre_allocate_vectors(update_input, update_target, update_pred, update_output,
                                  1, 1, prediction_len);
}

// This should match the behavior of sliding_windows() in the PyTorch version
std::pair<std::vector<std::vector<float>>, std::vector<float>> AnomalousThresholdGenerator::create_sliding_windows(const std::vector<float> &data, int lookback_len, int prediction_len) {
    std::vector<std::vector<float>> x;
    std::vector<float> y;
    
    // Make sure there's enough data
    if (data.size() < lookback_len + prediction_len) {
        return {x, y};
    }
    
    // Create sliding windows
    for (size_t i = 0; i <= data.size() - lookback_len - prediction_len; i++) {
        std::vector<float> window;
        for (int j = 0; j < lookback_len; j++) {
            window.push_back(data[i + j]);
        }
        x.push_back(window);
        
        // For the target, we take the next value after the window
        y.push_back(data[i + lookback_len]);
    }
    
    return {x, y};
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
AnomalousThresholdGenerator::train(int epoch, float lr, const std::vector<float>& data2learn, int window_size) {
    // Input validation
    if (data2learn.size() < window_size + prediction_len) {
        throw std::runtime_error("Not enough data for generator training");
    }
    
    // Create sliding windows with the explicit window_size
    auto windows = create_sliding_windows(data2learn, window_size, prediction_len);
    
    // Check if windows were successfully created
    if (windows.first.empty()) {
        throw std::runtime_error("Failed to create training windows");
    }
    
    // Set model to training mode
    generator->train();
    
    // Track training progress
    std::vector<float> loss_history;
    
    // Training loop
    for (int e = 0; e < epoch; ++e) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < windows.first.size(); ++i) {
            // Reshape input to match the expected format
            std::vector<std::vector<std::vector<float>>> reshaped_input(1);
            reshaped_input[0].resize(1);
            reshaped_input[0][0] = windows.first[i];
            
            // Target value
            std::vector<float> target{windows.second[i]};
            
            // Training step
            float sample_loss = generator->train_step(reshaped_input, target, lr);
            epoch_loss += sample_loss;
        }
        
        // Calculate average loss for this epoch
        float avg_loss = epoch_loss / static_cast<float>(windows.first.size());
        loss_history.push_back(avg_loss);
        
        // Report progress
        if ((e + 1) % 100 == 0 || e == 0) {
            std::cout << "Generator Epoch " << (e + 1) << "/" << epoch
                     << ", Average Loss: " << avg_loss << std::endl;
        }
        
        // Optional: Early stopping check
        if (loss_history.size() > 2 && 
            std::abs(loss_history[loss_history.size()-1] - loss_history[loss_history.size()-2]) < 1e-5) {
            std::cout << "Early stopping at epoch " << (e + 1) << ", loss stabilized" << std::endl;
            break;
        }
    }
    
    // Apply learning (finalize training)
    generator->learn();
    
    // Convert windows to the expected 3D format for return
    std::vector<std::vector<std::vector<float>>> x3d;
    for (const auto& window : windows.first) {
        std::vector<std::vector<float>> batch;
        batch.push_back(window);
        x3d.push_back(batch);
    }
    
    return {x3d, windows.second};
}

void AnomalousThresholdGenerator::update(int epoch_update, float lr_update,
                                       const std::vector<float>& past_errors, 
                                       float recent_error) {
    
    // Prepare input for forward pass
    std::vector<std::vector<std::vector<float>>> input_tensor(1);
    input_tensor[0].resize(1);
    input_tensor[0][0] = past_errors;
    
    // Target is a single value
    std::vector<float> target = {recent_error};
    
    // Fast update loop with early stopping
    std::vector<float> loss_history;
    
    for (int e = 0; e < epoch_update; ++e) {
        // Get loss directly from train_step
        float current_loss = generator->train_step(input_tensor, target, lr_update);
        
        // Early stopping logic - if loss increases, stop training
        if (!loss_history.empty() && current_loss > loss_history.back()) {
            break;
        }
        
        loss_history.push_back(current_loss);
    }
}

float AnomalousThresholdGenerator::generate(const std::vector<float> &prediction_errors,
                                           float minimal_threshold) {
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].resize(1);
    reshaped_input[0][0] = prediction_errors;

    std::vector<float> prediction = generator->forward(reshaped_input);

    float threshold = std::max(minimal_threshold, prediction[0]);

    return threshold;
}

void AnomalousThresholdGenerator::save_weights(std::ofstream &file) {
    if (generator) {
        generator->save_weights(file);
    } else {
        throw std::runtime_error("Save weights: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::save_biases(std::ofstream &file) {
    if (generator) {
        generator->save_biases(file);
    } else {
        throw std::runtime_error("Save biases: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::load_weights(std::ifstream &file) {
    if (generator) {
        generator->load_weights(file);
    } else {
        throw std::runtime_error("Load weights: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::load_biases(std::ifstream &file) {
    if (generator) {
        generator->load_biases(file);
    } else {
        throw std::runtime_error("Load biases: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::initialize_layer_cache() {
    if (generator) {
        generator->initialize_layer_cache();
    } else {
        throw std::runtime_error("Initialize layer cache: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::reset_states() {
    if (generator) {
        generator->reset_states();
    } else {
        throw std::runtime_error("Reset states: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::clear_update_state() {
    if (generator) {
        generator->clear_update_state();
    } else {
        throw std::runtime_error("Clear update state: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::save_model_state(std::ofstream &file) {
    if (generator) {
        generator->save_model_state(file);
    } else {
        throw std::runtime_error("Save model state: Generator not initialized");
    }
}

void AnomalousThresholdGenerator::load_model_state(std::ifstream &file) {
    if (generator) {
        generator->load_model_state(file);
    } else {
        throw std::runtime_error("Load model state: Generator not initialized");
    }
}


