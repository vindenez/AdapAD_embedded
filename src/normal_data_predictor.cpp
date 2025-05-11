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
NormalDataPredictor::create_sliding_windows(const std::vector<float> &data) {
    return ::create_sliding_windows(data, lookback_len, prediction_len);
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
NormalDataPredictor::train(int epoch, float lr, const std::vector<float> &data2learn) {
    std::cout << "Starting training with " << data2learn.size() << " samples..." << std::endl;
    auto windows = create_sliding_windows(data2learn);
    std::cout << "Created " << windows.first.size() << " training windows" << std::endl;
    
    predictor->train();
    float best_loss = std::numeric_limits<float>::max();
    int no_improve_count = 0;
    
    for (int e = 0; e < epoch; ++e) {
        float epoch_loss = 0.0f;
        for (size_t i = 0; i < windows.first.size(); ++i) {
            std::vector<std::vector<std::vector<float>>> input_tensor(1);
            input_tensor[0].resize(1);
            input_tensor[0][0] = windows.first[i];
            
            // Forward pass to get prediction for loss logging
            std::vector<float> pred = predictor->forward(input_tensor);
            
            // Calculate loss
            float sample_loss = 0.0f;
            std::vector<float> target{windows.second[i]};
            for (size_t k = 0; k < pred.size(); ++k) {
                float diff = pred[k] - target[k];
                sample_loss += diff * diff;
            }
            epoch_loss += sample_loss;
            
            // Train step for each sample - no need to pass output anymore
            predictor->train_step(input_tensor, target, lr);
        }
        
        // Report progress
        if ((e + 1) % 100 == 0) {
            float avg_loss = epoch_loss / static_cast<float>(windows.first.size());
            std::cout << "Epoch " << (e + 1) << "/" << epoch << ", Average Loss: " << avg_loss
                      << std::endl;
        }
    }
    
    predictor->learn();
    
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

float NormalDataPredictor::predict(const std::vector<std::vector<std::vector<float>>> &observed) {
    std::vector<float> prediction = predictor->forward(observed);
    return prediction[0];
}

void NormalDataPredictor::update(int epoch_update, float lr_update,
           const std::vector<std::vector<std::vector<float>>> &past_observations,
           const std::vector<float> &recent_observation) {
    
    for (int e = 0; e < epoch_update; ++e) {
        predictor->train_step(past_observations, recent_observation, lr_update);
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


