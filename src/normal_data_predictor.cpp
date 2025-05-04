#include "normal_data_predictor.hpp"
#include "normal_data_prediction_error_calculator.hpp"
#include "matrix_utils.hpp"
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sys/resource.h>

NormalDataPredictor::NormalDataPredictor(
    int lstm_layer, int lstm_unit, int lookback_len, int prediction_len)
    : lookback_len(lookback_len),
      prediction_len(prediction_len) {
      
    predictor = LSTMPredictorFactory::create_predictor(
        prediction_len,  // num_classes
        lookback_len,    // input_size
        lstm_unit,       // hidden_size
        lstm_layer,      // num_layers
        lookback_len,    // lookback_len
        true);           // batch_first

    // Initialize layer cache
    predictor->initialize_layer_cache();

    // Pre-allocate vectors for update
    update_input.resize(1);
    update_input[0].resize(1);
    update_input[0][0].resize(lookback_len, 0.0f);
    update_target.resize(prediction_len, 0.0f);
    update_pred.resize(prediction_len, 0.0f);
    update_output.sequence_output.resize(1);
    update_output.sequence_output[0].resize(1);
    update_output.sequence_output[0][0].resize(prediction_len, 0.0f);
    update_output.final_hidden.resize(lstm_layer);
    update_output.final_cell.resize(lstm_layer);
    for (int i = 0; i < lstm_layer; ++i) {
        update_output.final_hidden[i].resize(lstm_unit, 0.0f);
        update_output.final_cell[i].resize(lstm_unit, 0.0f);
    }
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
NormalDataPredictor::create_sliding_windows(const std::vector<float>& data) {
    return ::create_sliding_windows(data, lookback_len, prediction_len);
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
NormalDataPredictor::train(int epoch, float lr, const std::vector<float>& data2learn) {
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
            
            // Forward pass
            auto output = predictor->forward(input_tensor);
            auto pred = predictor->get_final_prediction(output);
            
            // Calculate loss
            float sample_loss = 0.0f;
            std::vector<float> target{windows.second[i]};
            for (size_t k = 0; k < pred.size(); ++k) {
                float diff = pred[k] - target[k];
                sample_loss += diff * diff;
            }
            
            epoch_loss += sample_loss;
            
            // Train step for each sample - now passing the output
            predictor->train_step(input_tensor, target, output, lr);
        }
        
        // Report progress
        if ((e + 1) % 100 == 0) {
            float avg_loss = epoch_loss / static_cast<float>(windows.first.size());
            std::cout << "Epoch " << (e + 1) << "/" << epoch 
                     << ", Average Loss: " << avg_loss << std::endl;
        }
    }

    predictor->learn();
    
    // Convert windows to 3D
    std::vector<std::vector<std::vector<float>>> x3d;
    for (const auto& window : windows.first) {
        std::vector<std::vector<std::vector<float>>> input_tensor(1);
        input_tensor[0].resize(1);
        input_tensor[0][0] = window;
        x3d.push_back(input_tensor[0]);
    }
    
    return {x3d, windows.second};
}

float NormalDataPredictor::predict(const std::vector<std::vector<std::vector<float>>>& observed) {
    bool was_training = predictor->is_training();  
    predictor->eval();  
    
    // Reshape input to match Python version: (batch=1, features=lookback_len)
    if (observed.size() != 1 || observed[0].size() != 1 || 
        observed[0][0].size() != lookback_len) {
        throw std::runtime_error("Invalid input dimensions for prediction");
    }
    
    // Flatten the input to match Python's reshape(1, -1)
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].resize(1);
    reshaped_input[0][0] = observed[0][0];  
    
    auto output = predictor->forward(reshaped_input);
    auto pred = predictor->get_final_prediction(output);
    float result = std::max(0.0f, pred[0]);
    
    if (was_training) {
        predictor->train();  
    }
    
    return result;
}
void NormalDataPredictor::update(int epoch_update, float lr_update,
                               const std::vector<std::vector<std::vector<float>>>& past_observations,
                               const std::vector<float>& recent_observation) {
    // Validate input dimensions
    if (past_observations.empty() || past_observations[0].empty() ||
        past_observations[0][0].size() != lookback_len) {
        throw std::runtime_error("Invalid past_observations dimensions in update");
    }
    if (recent_observation.empty()) {
        throw std::runtime_error("Empty recent_observation in update");
    }

    // Copy data to pre-allocated input
    std::copy(past_observations[0][0].begin(), past_observations[0][0].end(),
              update_input[0][0].begin());
    
    // Copy target to pre-allocated vector
    std::copy(recent_observation.begin(), recent_observation.end(),
              update_target.begin());
    
    predictor->train();  // Ensure training mode is enabled
    
    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        // Perform forward pass to get fresh output for this epoch
        auto output = predictor->forward(update_input);
        
        // Get prediction and compute loss
        auto pred = predictor->get_final_prediction(output);
        float current_loss = 0.0f;
        for (size_t i = 0; i < pred.size(); ++i) {
            float diff = pred[i] - update_target[i];
            current_loss += diff * diff;
        }
        // Update step using fresh forward pass output
        predictor->train_step(update_input, update_target, output, lr_update);
    }
}

void NormalDataPredictor::save_weights(std::ofstream& file) {
    if (predictor) {
        predictor->save_weights(file);
    } else {
        throw std::runtime_error("Predictor not initialized");
    }
}

void NormalDataPredictor::save_biases(std::ofstream& file) {
    if (predictor) {
        predictor->save_biases(file);
    } else {
        throw std::runtime_error("Predictor not initialized");
    }
}

void NormalDataPredictor::load_weights(std::ifstream& file) {
    if (predictor) {
        predictor->load_weights(file);
    } else {
        throw std::runtime_error("Predictor not initialized");
    }
}

void NormalDataPredictor::load_biases(std::ifstream& file) {
    if (predictor) {
        predictor->load_biases(file);
    } else {
        throw std::runtime_error("Predictor not initialized");
    }
}

void NormalDataPredictor::save_layer_cache(std::ofstream& file) const {
    predictor->save_layer_cache(file);
}

void NormalDataPredictor::load_layer_cache(std::ifstream& file) {
    predictor->load_layer_cache(file);
}

void NormalDataPredictor::initialize_layer_cache() {
    if (predictor) {
        predictor->initialize_layer_cache();
    } else {
        throw std::runtime_error("Predictor not initialized");
    }
}

void NormalDataPredictor::clear_update_state() {
    if (predictor) {
        predictor->clear_update_state();
    } else {
        throw std::runtime_error("Predictor not initialized");
    }
}

void NormalDataPredictor::reset_states() {
    if (predictor) {
        predictor->reset_states();
    } else {
        throw std::runtime_error("Predictor not initialized");
    }
}

