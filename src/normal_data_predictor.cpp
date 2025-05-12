#include "normal_data_predictor.hpp"
#include "config.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

NormalDataPredictor::NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len,
                                         int prediction_len)
    : lookback_len(lookback_len), prediction_len(prediction_len) {

    predictor = LSTMPredictorFactory::create_predictor(prediction_len, // num_classes
                                                       lookback_len,   // input_size
                                                       lstm_unit,      // hidden_size
                                                       lstm_layer,     // num_layers
                                                       lookback_len,   // lookback_len
                                                       true);          // batch_first

    // Initialize layer cache
    predictor->initialize_layer_cache();

    // Pre-allocate vectors using the new method
    predictor->pre_allocate_vectors(update_input, update_target, update_pred, update_output, 
                                  1, 1, prediction_len);
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
NormalDataPredictor::create_sliding_windows(const std::vector<float> &data, int lookback_len, int prediction_len) {
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
NormalDataPredictor::train(int epoch, float lr, const std::vector<float>& data2learn, int window_size) {
    // Input validation
    if (data2learn.size() < window_size + prediction_len) {
        throw std::runtime_error("Not enough data for predictor training");
    }
    
    // Create sliding windows with the explicit window_size
    auto windows = create_sliding_windows(data2learn, window_size, prediction_len);
    
    // Check if windows were successfully created
    if (windows.first.empty()) {
        throw std::runtime_error("Failed to create training windows");
    }
    
    // Set model to training mode
    predictor->train();
    
    // Early stopping parameters
    const int patience = 10;  // Number of epochs to wait for improvement
    const float min_delta = 1e-4;  // Minimum change to qualify as improvement
    float best_loss = std::numeric_limits<float>::max();
    int no_improvement_count = 0;
    
    // Training loop
    for (int e = 0; e < epoch; ++e) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < windows.first.size(); ++i) {
            // Reshape input
            std::vector<std::vector<std::vector<float>>> reshaped_input(1);
            reshaped_input[0].resize(1);
            reshaped_input[0][0] = windows.first[i];
            
            // Target value
            std::vector<float> target{windows.second[i]};
            
            // Training step
            float sample_loss = predictor->train_step(reshaped_input, target, lr);
            epoch_loss += sample_loss;
        }
        
        // Calculate average loss for this epoch
        float avg_loss = epoch_loss / static_cast<float>(windows.first.size());
        
        // Report progress
        if ((e + 1) % 50 == 0 || e == 0) {
            std::cout << "Predictor Epoch " << (e + 1) << "/" << epoch 
                     << ", Average Loss: " << avg_loss << std::endl;
        }
        
        // Early stopping check
        if (avg_loss < best_loss - min_delta) {
            // Loss improved
            best_loss = avg_loss;
            no_improvement_count = 0;
        } else {
            // Loss didn't improve
            no_improvement_count++;
            if (no_improvement_count >= patience) {
                std::cout << "Early stopping at epoch " << (e + 1) 
                         << " as loss didn't improve for " << patience << " epochs" << std::endl;
                break;
            }
        }
        
        // Optional: Exit if loss is very small
        if (avg_loss < 1e-6) {
            std::cout << "Early stopping at epoch " << (e + 1) 
                     << " as loss is very small (" << avg_loss << ")" << std::endl;
            break;
        }
    }
    
    // Apply learning (finalize training)
    predictor->learn();
    
    // Convert windows to the expected 3D format for return
    std::vector<std::vector<std::vector<float>>> x3d;
    for (const auto& window : windows.first) {
        std::vector<std::vector<float>> batch;
        batch.push_back(window);
        x3d.push_back(batch);
    }
    
    return {x3d, windows.second};
}

float NormalDataPredictor::predict(const std::vector<std::vector<std::vector<float>>> &observed) {
    std::vector<float> prediction = predictor->forward(observed);
    return prediction[0];
}

void NormalDataPredictor::update(int epoch_update, float lr_update,
                               const std::vector<std::vector<std::vector<float>>> &past_observations,
                               const std::vector<float> &recent_observation) {
    
    // Fast update loop with early stopping
    std::vector<float> loss_history;
    
    for (int e = 0; e < epoch_update; ++e) {
        // Get loss directly from train_step
        float current_loss = predictor->train_step(past_observations, recent_observation, lr_update);
        
        // Early stopping logic - if loss increases, stop training
        if (!loss_history.empty() && current_loss > loss_history.back()) {
            break;
        }
        
        loss_history.push_back(current_loss);
    }
}

void NormalDataPredictor::save_weights(std::ofstream &file) {
    if (predictor) {
        predictor->save_weights(file);
    } else {
        throw std::runtime_error("Save weights: Predictor not initialized");
    }
}

void NormalDataPredictor::save_biases(std::ofstream &file) {
    if (predictor) {
        predictor->save_biases(file);
    } else {
        throw std::runtime_error("Save biases: Predictor not initialized");
    }
}

void NormalDataPredictor::load_weights(std::ifstream &file) {
    if (predictor) {
        predictor->load_weights(file);
    } else {
        throw std::runtime_error("Load weights: Predictor not initialized");
    }
}

void NormalDataPredictor::load_biases(std::ifstream &file) {
    if (predictor) {
        predictor->load_biases(file);
    } else {
        throw std::runtime_error("Load biases: Predictor not initialized");
    }
}

void NormalDataPredictor::initialize_layer_cache() {
    if (predictor) {
        predictor->initialize_layer_cache();
    } else {
        throw std::runtime_error("Initialize layer cache: Predictor not initialized");
    }
}

void NormalDataPredictor::clear_update_state() {
    if (predictor) {
        predictor->clear_update_state();
    } else {
        throw std::runtime_error("Clear update state: Predictor not initialized");
    }
}

void NormalDataPredictor::reset_states() {
    if (predictor) {
        predictor->reset_states();
    } else {
        throw std::runtime_error("Reset states: Predictor not initialized");
    }
}

void NormalDataPredictor::save_model_state(std::ofstream &file) {
    if (predictor) {
        predictor->save_model_state(file);
    } else {
        throw std::runtime_error("Save model state: Predictor not initialized");
    }
}

void NormalDataPredictor::load_model_state(std::ifstream &file) {
    if (predictor) {
        predictor->load_model_state(file);
    } else {
        throw std::runtime_error("Load model state: Predictor not initialized");
    }
}


