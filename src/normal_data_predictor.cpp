#include "normal_data_predictor.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "matrix_utils.hpp"
#include <chrono>

NormalDataPredictor::NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len)
    : num_layers(lstm_layer), 
      hidden_size(lstm_unit), 
      lookback_len(lookback_len), 
      prediction_len(prediction_len),
      predictor(prediction_len,    // num_classes
               lookback_len,      // input_size
               lstm_unit,         // hidden_size
               lstm_layer,        // num_layers
               lookback_len) {    // lookback_len
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> NormalDataPredictor::train(int num_epochs, float learning_rate, const std::vector<float>& data_to_learn, const EarlyStoppingCallback& callback) {
    std::cout << "\n=== Training Data Debug ===" << std::endl;
    std::cout << "First 5 training values (should be normalized [0,1]):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), data_to_learn.size()); ++i) {
        std::cout << "  [" << i << "]: " << data_to_learn[i] << std::endl;
    }
    
    auto [x, y] = sliding_windows(data_to_learn, lookback_len, prediction_len);
    
    std::cout << "\nFirst 3 training sequences:" << std::endl;
    for (size_t seq = 0; seq < std::min(size_t(3), x.size()); ++seq) {
        std::cout << "\nSequence " << seq << ":" << std::endl;
        std::cout << "Input (x):" << std::endl;
        for (size_t i = 0; i < x[seq].size(); ++i) {
            std::cout << "  [" << i << "]: " << x[seq][i] << std::endl;
        }
        std::cout << "Target (y):" << std::endl;
        std::cout << "  [0]: " << y[seq][0] << std::endl;
    }
    
    predictor.train();
    predictor.init_adam_optimizer(learning_rate);

    std::cout << "\nTraining on " << x.size() << " sequences for " << num_epochs << " epochs..." << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            predictor.zero_grad();
            
            std::vector<std::vector<std::vector<float>>> reshaped_input(1);
            reshaped_input[0].push_back(x[i]);
            
            auto outputs = predictor.forward(reshaped_input);
            
            if (epoch % 100 == 0 && i < 3) {
                std::cout << "Sample prediction at epoch " << epoch << " sequence " << i << ":" << std::endl;
                std::cout << "  Target: " << y[i][0] << std::endl;
                std::cout << "  Predicted: " << outputs[0] << std::endl;
            }
            
            float loss = compute_mse_loss(outputs, y[i]);
            
            predictor.backward(y[i], "MSE");
            predictor.update_parameters_adam(learning_rate);
            
            epoch_loss += loss;
        }
        
        epoch_loss /= x.size();
        
        if (epoch % 100 == 0) {
            std::cout << "Training Epoch " << epoch << "/" << num_epochs 
                     << " Loss: " << std::scientific << epoch_loss << std::endl;
        }
    }

    predictor.eval();

    return {x, y};
}

float NormalDataPredictor::predict(const std::vector<float>& observed) {
    predictor.eval();
    
    if (observed.size() < lookback_len) {
        throw std::runtime_error("Not enough observations for prediction");
    }
    
    std::vector<float> input_sequence(observed.end() - lookback_len, observed.end());
    
    std::cout << "\n=== Prediction Debug ===" << std::endl;
    std::cout << "Input sequence (should be normalized [0,1]):" << std::endl;
    for (size_t i = 0; i < input_sequence.size(); ++i) {
        std::cout << "  [" << i << "]: " << input_sequence[i] << std::endl;
    }
    
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].push_back(input_sequence);
    
    auto predicted_val = predictor.forward(reshaped_input);
    
    // Match Python: only clamp lower bound at 0
    float clamped_prediction = std::max(0.0f, predicted_val[0]);
    
    std::cout << "LSTM output (raw): " << predicted_val[0] << std::endl;
    std::cout << "LSTM output (only lower bound clamped): " << clamped_prediction << std::endl;
    
    return clamped_prediction;
}

void NormalDataPredictor::update(int epoch_update, float lr_update, 
                                const std::vector<float>& past_observations, 
                                float recent_observation) {
    predictor.train();
    predictor.init_adam_optimizer(lr_update);
    
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].push_back(past_observations);

    // Convert single float to vector for LSTM
    std::vector<float> target_vec = {recent_observation};
    std::vector<float> loss_history;  // Track loss history for early stopping

    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        auto predicted_val = predictor.forward(reshaped_input);
        float loss = compute_mse_loss(predicted_val, target_vec);
        
        predictor.zero_grad();
        predictor.backward(target_vec, "MSE");
        predictor.update_parameters_adam(lr_update);

        // Early stopping like in Python
        if (!loss_history.empty() && loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(loss);
    }

    predictor.eval();
}



