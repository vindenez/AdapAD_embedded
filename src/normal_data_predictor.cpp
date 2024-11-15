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

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
NormalDataPredictor::train(int num_epochs, float learning_rate, 
                          const std::vector<float>& data_to_learn, 
                          const EarlyStoppingCallback& callback) {
    auto [x, y] = sliding_windows(data_to_learn, lookback_len, prediction_len);
    
    predictor.train();
    predictor.init_adam_optimizer();
    
    // Gradient check on first batch before training
    std::cout << "\n=== Gradient Check for Data Predictor ===" << std::endl;
    std::vector<std::vector<std::vector<float>>> first_input;
    reshape_input(x[0], first_input);
    bool gradients_ok = predictor.check_gradients(first_input, y[0]);
    std::cout << "Gradient check " << (gradients_ok ? "PASSED" : "FAILED") << "\n" << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            predictor.zero_grad();
            
            std::vector<std::vector<std::vector<float>>> reshaped_input;
            reshape_input(x[i], reshaped_input);
            
            auto outputs = predictor.forward(reshaped_input);
            float loss = compute_mse_loss(outputs, y[i]);
            
            predictor.backward(y[i], "MSE");
            predictor.update_parameters_adam();
            
            epoch_loss += loss;
        }
        
        epoch_loss /= x.size();
    }

    predictor.eval();
    return {x, y};
}

float NormalDataPredictor::predict(const std::vector<float>& observed) {
    predictor.eval();
    
    std::vector<std::vector<std::vector<float>>> reshaped_input;
    reshape_input(observed, reshaped_input);
    
    predictor.reset_states();
    std::vector<float> output = predictor.forward(reshaped_input);
    
    if (output.empty()) {
        throw std::runtime_error("Empty output from predictor");
    }
    
    float predicted_val = std::max(0.0f, output[0]);
    return predicted_val;
}

void NormalDataPredictor::update(int epoch_update, float lr_update, 
                                const std::vector<float>& past_observations, 
                                float recent_observation) {
    predictor.train();
    predictor.init_adam_optimizer();
    
    std::vector<std::vector<std::vector<float>>> reshaped_input;
    reshape_input(past_observations, reshaped_input);

    std::vector<float> target_vec = {recent_observation};
    std::vector<float> loss_history;

    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        predictor.reset_states();
        auto predicted_val = predictor.forward(reshaped_input);
        float loss = compute_mse_loss(predicted_val, target_vec);
        
        predictor.zero_grad();
        predictor.backward(target_vec, "MSE");
        predictor.update_parameters_adam();

        if (!loss_history.empty() && loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(loss);
    }

    predictor.eval();
}

void NormalDataPredictor::reshape_input(const std::vector<float>& input_sequence, 
                                    std::vector<std::vector<std::vector<float>>>& reshaped) {
    if (input_sequence.size() < lookback_len) {
        throw std::runtime_error("Input sequence length is less than lookback_len");
    }

    // Reshape to [batch_size=1, seq_len=1, input_size=lookback_len]
    reshaped.clear();
    reshaped.resize(1);        // batch_size = 1
    reshaped[0].resize(1);     // seq_len = 1
    reshaped[0][0].resize(lookback_len);  // input_size = lookback_len
    
    // Copy the last lookback_len values
    for (int i = 0; i < lookback_len; i++) {
        reshaped[0][0][i] = input_sequence[input_sequence.size() - lookback_len + i];
    }
}


