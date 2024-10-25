#include "normal_data_predictor.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "matrix_utils.hpp"

NormalDataPredictor::NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len)
    : num_layers(lstm_layer), 
      hidden_size(lstm_unit), 
      lookback_len(lookback_len), 
      prediction_len(prediction_len),
      predictor(lookback_len, hidden_size, prediction_len, num_layers, lookback_len) {
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> NormalDataPredictor::train(int num_epochs, float learning_rate, const std::vector<float>& data_to_learn) {
    auto [x, y] = sliding_windows(data_to_learn, lookback_len, prediction_len);
    
    predictor.train();
    predictor.init_adam_optimizer(learning_rate);
    std::vector<float> loss_history;

    std::cout << "\nTraining for " << num_epochs << " epochs..." << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            predictor.zero_grad();
            auto outputs = predictor.forward(x[i]);
            float loss = compute_mse_loss(outputs, y[i]);
            predictor.backward(y[i], "MSE");
            predictor.update_parameters_adam(learning_rate);
            epoch_loss += loss;
        }
        
        epoch_loss /= x.size();
        loss_history.push_back(epoch_loss);

        // Log progress every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Training Epoch " << epoch << "/" << num_epochs 
                     << " Loss: " << std::scientific << epoch_loss << std::endl;
        }

        // Early stopping with patience
        if (loss_history.size() > 50) {  // Increased from 10 to 50 epochs
            bool has_improved = false;
            float min_loss = loss_history[loss_history.size() - 50];
            for (size_t i = loss_history.size() - 49; i < loss_history.size(); ++i) {
                if (loss_history[i] < min_loss * 0.9999f) {  // More lenient improvement threshold
                    has_improved = true;
                    break;
                }
            }
            if (!has_improved) {
                std::cout << "Early stopping at epoch " << epoch << " with loss " << epoch_loss << std::endl;
                break;
            }
        }
    }

    return {x, y};
}

std::vector<float> NormalDataPredictor::predict(const std::vector<float>& input) {
    predictor.eval();
    auto predicted_val = predictor.forward(input);
    predicted_val[0] = std::max(0.0f, predicted_val[0]);
    return predicted_val;
}

void NormalDataPredictor::update(int epoch_update, float lr_update, const std::vector<float>& past_observations, const std::vector<float>& recent_observation) {
    predictor.train();
    predictor.init_adam_optimizer(lr_update);
    std::vector<float> loss_history;

    for (int epoch = 0; epoch < epoch_update; ++epoch) {
        auto predicted_val = predictor.forward(past_observations);
        
        float loss = compute_mse_loss(predicted_val, recent_observation);
        
        predictor.zero_grad();
        predictor.backward(recent_observation, "MSE");
        predictor.update_parameters_adam(lr_update);

        loss_history.push_back(loss);

        if (loss_history.size() > 1 && loss > loss_history[loss_history.size() - 2]) {
            break;
        }

    }
}



