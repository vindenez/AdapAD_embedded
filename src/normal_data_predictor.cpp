#include "normal_data_predictor.hpp"
#include "normal_data_prediction_error_calculator.hpp"
#include "matrix_utils.hpp"
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

NormalDataPredictor::NormalDataPredictor(int lstm_layer, int lstm_unit, 
                                       int lookback_len, int prediction_len)
    : lookback_len(lookback_len),
      prediction_len(prediction_len) {
    
    predictor.reset(new LSTMPredictor(
        prediction_len,  // num_classes
        lookback_len,    // input_size
        lstm_unit,       // hidden_size
        lstm_layer,      // num_layers
        lookback_len     // seq_length
    ));
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

    predictor->train();
    
    std::vector<float> loss_l;  
    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        auto output = predictor->forward(past_observations);
        auto pred = predictor->get_final_prediction(output);
        
        // Calculate MSE loss
        float current_loss = 0.0f;
        for (size_t i = 0; i < pred.size(); ++i) {
            float diff = pred[i] - recent_observation[i];
            current_loss += diff * diff;
        }
        current_loss /= static_cast<float>(pred.size());
        
        // Early stopping
        if (!loss_l.empty() && current_loss > loss_l.back()) {  
            break;
        }
        
        loss_l.push_back(current_loss);
        predictor->train_step(past_observations, recent_observation, output, lr_update);
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

