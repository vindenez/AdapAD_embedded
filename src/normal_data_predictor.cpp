#include "normal_data_predictor.hpp"
#include <stdexcept>

NormalDataPredictor::NormalDataPredictor(const std::unordered_map<std::string, std::vector<std::vector<float>>>& weights,
                                         const std::unordered_map<std::string, std::vector<float>>& biases) {
    // Load all LSTM layers
    for (int i = 0;; ++i) {
        std::string weight_ih_key = "lstm.weight_ih_l" + std::to_string(i);
        std::string weight_hh_key = "lstm.weight_hh_l" + std::to_string(i);
        std::string bias_ih_key = "lstm.bias_ih_l" + std::to_string(i);
        std::string bias_hh_key = "lstm.bias_hh_l" + std::to_string(i);

        if (weights.find(weight_ih_key) == weights.end()) {
            break; // No more LSTM layers
        }

        lstm_layers.emplace_back(weights.at(weight_ih_key), weights.at(weight_hh_key),
                                 biases.at(bias_ih_key), biases.at(bias_hh_key));
    }

    // Load the fully connected layer
    fc_weight = weights.at("fc.weight");
    fc_bias = biases.at("fc.bias");
}

std::vector<float> NormalDataPredictor::predict(const std::vector<float>& input) {
    std::vector<float> output = input;
    // Forward pass through all LSTM layers
    for (auto& lstm : lstm_layers) {
        output = lstm.forward(output);
    }
    // Pass through the fully connected layer
    output = fully_connected(output);
    return output;
}

std::vector<float> NormalDataPredictor::fully_connected(const std::vector<float>& input) {
    std::vector<float> result(fc_weight.size(), 0.0f);
    for (size_t i = 0; i < fc_weight.size(); ++i) {
        for (size_t j = 0; j < fc_weight[i].size(); ++j) {
            result[i] += fc_weight[i][j] * input[j];
        }
        result[i] += fc_bias[i];
    }
    return result;
}
