#include "normal_data_predictor.hpp"
#include <iostream>
#include <stdexcept>

// Constructor to initialize LSTM layers and the fully connected layer
NormalDataPredictor::NormalDataPredictor(const std::unordered_map<std::string, std::vector<std::vector<float>>>& weights,
                                         const std::unordered_map<std::string, std::vector<float>>& biases) {
    // Load all LSTM layers
    for (int i = 0;; ++i) {
        std::string weight_ih_key = "lstm.weight_ih_l" + std::to_string(i);
        std::string weight_hh_key = "lstm.weight_hh_l" + std::to_string(i);
        std::string bias_ih_key = "lstm.bias_ih_l" + std::to_string(i);
        std::string bias_hh_key = "lstm.bias_hh_l" + std::to_string(i);

        // Break if the LSTM layer weights and biases are not found
        if (weights.find(weight_ih_key) == weights.end() || weights.find(weight_hh_key) == weights.end() ||
            biases.find(bias_ih_key) == biases.end() || biases.find(bias_hh_key) == biases.end()) {
            break; // No more LSTM layers
        }

        // Debugging weights and biases for each LSTM layer
        std::cout << "Loading LSTM Layer " << i << "..." << std::endl;
        std::cout << "Weight_ih dimensions: " << weights.at(weight_ih_key).size() << " x " 
                  << (weights.at(weight_ih_key).empty() ? 0 : weights.at(weight_ih_key)[0].size()) << std::endl;
        std::cout << "Weight_hh dimensions: " << weights.at(weight_hh_key).size() << " x " 
                  << (weights.at(weight_hh_key).empty() ? 0 : weights.at(weight_hh_key)[0].size()) << std::endl;
        std::cout << "Bias_ih size: " << biases.at(bias_ih_key).size() << std::endl;
        std::cout << "Bias_hh size: " << biases.at(bias_hh_key).size() << std::endl;

        // Add the LSTM layer using the consolidated weights
        lstm_layers.emplace_back(weights.at(weight_ih_key), weights.at(weight_hh_key),
                                 biases.at(bias_ih_key), biases.at(bias_hh_key));
    }

    // Initialize input size based on the first LSTM layer
    if (!lstm_layers.empty()) {
        input_size = lstm_layers[0].get_input_size(); // Assuming get_input_size() returns a valid input size
        if (input_size <= 0) {
            throw std::runtime_error("Error: Input size for LSTM layer cannot be zero or negative.");
        }
    } else {
        throw std::runtime_error("No LSTM layers found in the provided weights.");
    }

    // Load the fully connected layer
    std::string fc_weight_key = "fc.weight";
    std::string fc_bias_key = "fc.bias";

    if (weights.find(fc_weight_key) != weights.end() && biases.find(fc_bias_key) != biases.end()) {
        fc_weight = weights.at(fc_weight_key);
        fc_bias = biases.at(fc_bias_key);

        // Debugging the fully connected layer weights and biases
        std::cout << "Fully Connected Layer Weight dimensions: " << fc_weight.size() << " x " 
                  << (fc_weight.empty() ? 0 : fc_weight[0].size()) << std::endl;
        std::cout << "Fully Connected Layer Bias size: " << fc_bias.size() << std::endl;
    } else {
        throw std::runtime_error("Fully connected layer weights or biases not found.");
    }
}


// Method to get input size
int NormalDataPredictor::get_input_size() const {
    return input_size;
}

// Predict function with forward pass through LSTM layers and fully connected layer
std::vector<float> NormalDataPredictor::predict(const std::vector<float>& input) {
    // Check if input size matches expected input size
    if (input.size() != input_size) {
        std::cout << "Error: Input size does not match expected input size (" << input.size() << " vs " << input_size << ")." << std::endl;
        return {};
    }

    std::vector<float> output = input;

    // Forward pass through all LSTM layers
    for (size_t i = 0; i < lstm_layers.size(); ++i) {
        std::cout << "Forward pass through LSTM layer " << i << " with input size: " << output.size() << std::endl;
        output = lstm_layers[i].forward(output);

        // Debugging output size after each layer
        if (!output.empty()) {
            std::cout << "Output size after LSTM layer " << i << ": " << output.size() << std::endl;
        } else {
            std::cout << "Error: LSTM layer " << i << " forward pass returned an empty vector." << std::endl;
            return {};
        }
    }

    // Pass through the fully connected layer
    std::cout << "Passing through the fully connected layer with input size: " << output.size() << std::endl;
    output = fully_connected(output);

    // Debugging fully connected layer output
    if (!output.empty()) {
        std::cout << "Output size after fully connected layer: " << output.size() << std::endl;
    } else {
        std::cout << "Error: Fully connected layer forward pass returned an empty vector." << std::endl;
    }

    return output;
}

// Fully connected layer implementation
std::vector<float> NormalDataPredictor::fully_connected(const std::vector<float>& input) {
    if (input.size() != fc_weight[0].size()) {
        std::cout << "Error: Input size does not match fully connected layer input size (" << input.size() << " vs " << fc_weight[0].size() << ")." << std::endl;
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
