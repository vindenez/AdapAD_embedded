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

std::pair<std::vector<std::vector<float>>, std::vector<float>>
AnomalousThresholdGenerator::create_sliding_windows(const std::vector<float> &data) {
    return ::create_sliding_windows(data, lookback_len, prediction_len);
}

float AnomalousThresholdGenerator::generate(const std::vector<float> &prediction_errors,
                                            float minimal_threshold) {

    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].resize(1);
    reshaped_input[0][0] = prediction_errors;

    std::vector<float> pred = generator->forward(reshaped_input);

    // Clamp to minimal_threshold
    const auto &config = Config::getInstance();
    float result = std::max(minimal_threshold, pred[0] * config.threshold_multiplier);

    return result;
}

void AnomalousThresholdGenerator::update(int epoch_update, float lr_update,
                                         const std::vector<float> &past_errors,
                                         float recent_error) {

    // Copy data to pre-allocated input
    std::copy(past_errors.begin(), past_errors.end(), update_input[0][0].begin());

    // Set target
    update_target[0] = recent_error;

    for (int e = 0; e < epoch_update; ++e) {
        generator->train_step(update_input, update_target, lr_update);
    }
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
AnomalousThresholdGenerator::train(int epoch, float lr, const std::vector<float> &data2learn) {
    auto windows = create_sliding_windows(data2learn);
    
    generator->train();
    float best_loss = std::numeric_limits<float>::max();
    
    for (int e = 0; e < epoch; ++e) {
        float epoch_loss = 0.0f;
        for (size_t i = 0; i < windows.first.size(); ++i) {
            std::vector<std::vector<std::vector<float>>> input_tensor(1);
            input_tensor[0].resize(1);
            input_tensor[0][0] = windows.first[i];
            
            // Use single forward pass to get prediction
            std::vector<float> pred = generator->forward(input_tensor);
            
            // Calculate loss
            float sample_loss = 0.0f;
            std::vector<float> target{windows.second[i]};
            for (size_t k = 0; k < pred.size(); ++k) {
                float diff = pred[k] - target[k];
                sample_loss += diff * diff;
            }
            epoch_loss += sample_loss;
            
            // Train step for each sample - no need to pass output anymore
            generator->train_step(input_tensor, target, lr);
        }
        
        // Report progress periodically if needed
        if ((e + 1) % 100 == 0) {
            float avg_loss = epoch_loss / static_cast<float>(windows.first.size());
            std::cout << "Generator Epoch " << (e + 1) << "/" << epoch 
                      << ", Average Loss: " << avg_loss << std::endl;
        }
    }
    
    generator->learn();
    
    // Convert windows to 3D
    std::vector<std::vector<std::vector<float>>> x3d;
    for (const auto &window : windows.first) {
        std::vector<std::vector<std::vector<float>>> input_tensor(1);
        input_tensor[0].resize(1);
        input_tensor[0][0] = window;
        x3d.push_back(input_tensor[0]);
    }
    
    return {x3d, windows.second};
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


