#include "normal_data_predictor.hpp"
#include <iostream>
#include <stdexcept>

// Constructor to initialize LSTM layers and the fully connected layer
NormalDataPredictor::NormalDataPredictor(const std::unordered_map<std::string, std::vector<std::vector<float>>>& weights,
                                         const std::unordered_map<std::string, std::vector<float>>& biases) {
    // Load all LSTM layers
    for (int i = 0; i < 3; ++i) {  // We know there are 3 layers
        std::string weight_ih_key = "lstm.weight_ih_l" + std::to_string(i);
        std::string weight_hh_key = "lstm.weight_hh_l" + std::to_string(i);
        std::string bias_ih_key = "lstm.bias_ih_l" + std::to_string(i);
        std::string bias_hh_key = "lstm.bias_hh_l" + std::to_string(i);

        if (weights.find(weight_ih_key) == weights.end() || weights.find(weight_hh_key) == weights.end() ||
            biases.find(bias_ih_key) == biases.end() || biases.find(bias_hh_key) == biases.end()) {
            throw std::runtime_error("Missing weights or biases for LSTM layer " + std::to_string(i));
        }

        const auto& weight_ih = weights.at(weight_ih_key);
        const auto& weight_hh = weights.at(weight_hh_key);
        const auto& bias_ih = biases.at(bias_ih_key);
        const auto& bias_hh = biases.at(bias_hh_key);

        if (weight_ih.empty() || weight_hh.empty()) {
            throw std::runtime_error("Empty weight matrices for LSTM layer " + std::to_string(i));
        }

        int input_size = (i == 0) ? weight_ih[0].size() : lstm_layers[i-1].get_hidden_size();
        int hidden_size = weight_hh[0].size();  // Remove the division by 4

        lstm_layers.emplace_back(weight_ih, weight_hh, bias_ih, bias_hh, input_size, hidden_size);
    }

    // Load fully connected layer
    if (weights.find("fc.weight") != weights.end() && biases.find("fc.bias") != biases.end()) {
        fc_weight = weights.at("fc.weight");
        fc_bias = biases.at("fc.bias");
    } else {
        throw std::runtime_error("Missing fully connected layer weights/biases");
    }

    // Set input size based on the first LSTM layer
    input_size = weights.at("lstm.weight_ih_l0")[0].size();
}

// Method to get input size
int NormalDataPredictor::get_input_size() const {
    return input_size;
}

// Predict function with forward pass through LSTM layers and fully connected layer
std::vector<float> NormalDataPredictor::predict(const std::vector<float>& input) {
    std::vector<float> output = input;
    std::vector<float> h(lstm_layers[0].get_hidden_size(), 0.0f);
    std::vector<float> c(lstm_layers[0].get_hidden_size(), 0.0f);

    for (size_t i = 0; i < lstm_layers.size(); ++i) {
        // Ensure output size matches the expected input size for the current layer
        if (output.size() != lstm_layers[i].get_input_size()) {
            std::cerr << "Error: Mismatched input size for LSTM layer " << i << ". Expected: "
                      << lstm_layers[i].get_input_size() << ", Got: " << output.size() << std::endl;
            return {};
        }

        std::tie(output, h, c) = lstm_layers[i].forward(output, h, c);
        if (output.empty()) {
            throw std::runtime_error("Error: LSTM layer " + std::to_string(i) + " forward pass returned an empty vector.");
        }
    }

    output = fully_connected(output);

    return output;
}

// Fully connected layer implementation
std::vector<float> NormalDataPredictor::fully_connected(const std::vector<float>& input) {
    if (input.size() != fc_weight[0].size()) {
        std::cout << "Error: Input size does not match fully connected layer input size (" 
                  << input.size() << " vs " << fc_weight[0].size() << ")." << std::endl;
        return {};
    }

    std::vector<float> result(fc_weight.size(), 0.0f);
    for (size_t i = 0; i < fc_weight.size(); ++i) {
        for (size_t j = 0; j < fc_weight[i].size(); ++j) {
            result[i] += fc_weight[i][j] * input[j];
        }
        result[i] += fc_bias[i];
    }
    return result;
}