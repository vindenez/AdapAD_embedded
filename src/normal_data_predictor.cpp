#include "normal_data_predictor.hpp"
#include <iostream>
#include <stdexcept>

// Function to split concatenated weights into individual gate weights
std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
split_weights(const std::vector<std::vector<float>>& weights) {
    int total_rows = weights.size();
    int gate_size = total_rows / 4;  // Assuming 4 gates

    // Ensure that total_rows is divisible by 4
    if (total_rows % 4 != 0) {
        throw std::runtime_error("Total number of rows in weights is not divisible by 4");
    }

    // Split the weights along the rows
    std::vector<std::vector<float>> weight_input(weights.begin(), weights.begin() + gate_size);
    std::vector<std::vector<float>> weight_forget(weights.begin() + gate_size, weights.begin() + 2 * gate_size);
    std::vector<std::vector<float>> weight_cell(weights.begin() + 2 * gate_size, weights.begin() + 3 * gate_size);
    std::vector<std::vector<float>> weight_output(weights.begin() + 3 * gate_size, weights.end());

    return {weight_input, weight_forget, weight_cell, weight_output};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
split_biases(const std::vector<float>& biases) {
    int total_size = biases.size();
    int gate_size = total_size / 4;  // Assuming 4 gates

    // Ensure that total_size is divisible by 4
    if (total_size % 4 != 0) {
        throw std::runtime_error("Total size of biases is not divisible by 4");
    }

    // Split the biases
    std::vector<float> bias_input(biases.begin(), biases.begin() + gate_size);
    std::vector<float> bias_forget(biases.begin() + gate_size, biases.begin() + 2 * gate_size);
    std::vector<float> bias_cell(biases.begin() + 2 * gate_size, biases.begin() + 3 * gate_size);
    std::vector<float> bias_output(biases.begin() + 3 * gate_size, biases.end());

    return {bias_input, bias_forget, bias_cell, bias_output};
}

// Constructor to initialize LSTM layers and the fully connected layer
NormalDataPredictor::NormalDataPredictor(
    const std::unordered_map<std::string, std::vector<std::vector<float>>>& weights,
    const std::unordered_map<std::string, std::vector<float>>& biases) {

    // Load all LSTM layers
    for (int i = 0; i < 3; ++i) {  // Assuming 3 layers
        // Define keys for each layer
        std::string weight_ih_key = "lstm.weight_ih_l" + std::to_string(i);
        std::string weight_hh_key = "lstm.weight_hh_l" + std::to_string(i);
        std::string bias_ih_key = "lstm.bias_ih_l" + std::to_string(i);
        std::string bias_hh_key = "lstm.bias_hh_l" + std::to_string(i);

        // Check if the required keys exist
        if (weights.find(weight_ih_key) == weights.end() ||
            weights.find(weight_hh_key) == weights.end() ||
            biases.find(bias_ih_key) == biases.end() ||
            biases.find(bias_hh_key) == biases.end()) {
            throw std::runtime_error("Missing weights or biases for LSTM layer " + std::to_string(i));
        }

        // Retrieve concatenated weights and biases
        const auto& weight_ih_concatenated = weights.at(weight_ih_key);
        const auto& weight_hh_concatenated = weights.at(weight_hh_key);
        const auto& bias_ih_concatenated = biases.at(bias_ih_key);
        const auto& bias_hh_concatenated = biases.at(bias_hh_key);

        // Split concatenated weights and biases into individual gates
        auto [weight_ih_input, weight_ih_forget, weight_ih_cell, weight_ih_output] = split_weights(weight_ih_concatenated);
        auto [weight_hh_input, weight_hh_forget, weight_hh_cell, weight_hh_output] = split_weights(weight_hh_concatenated);
        auto [bias_ih_input, bias_ih_forget, bias_ih_cell, bias_ih_output] = split_biases(bias_ih_concatenated);
        auto [bias_hh_input, bias_hh_forget, bias_hh_cell, bias_hh_output] = split_biases(bias_hh_concatenated);

        // Determine input size and hidden size
        int layer_input_size = (i == 0) ? weight_ih_input[0].size() : lstm_layers.back().get_hidden_size();
        int layer_hidden_size = weight_hh_input.size();

        // Create LSTMPredictor instance for the current layer
        lstm_layers.emplace_back(
            weight_ih_input, weight_hh_input, bias_ih_input, bias_hh_input,
            weight_ih_forget, weight_hh_forget, bias_ih_forget, bias_hh_forget,
            weight_ih_output, weight_hh_output, bias_ih_output, bias_hh_output,
            weight_ih_cell, weight_hh_cell, bias_ih_cell, bias_hh_cell,
            layer_input_size, layer_hidden_size
        );
    }

    // Load fully connected layer
    if (weights.find("fc.weight") != weights.end() && biases.find("fc.bias") != biases.end()) {
        fc_weight = weights.at("fc.weight");
        fc_bias = biases.at("fc.bias");
    } else {
        throw std::runtime_error("Missing fully connected layer weights/biases");
    }

    // Set input size based on the first LSTM layer
    input_size = weights.at("lstm.weight_ih_l0")[0].size();  // Divide by 4 because weights are concatenated for 4 gates

    // Print information for debugging
    std::cout << "NormalDataPredictor initialized with input size: " << input_size << std::endl;
}

// Method to get input size
int NormalDataPredictor::get_input_size() const {
    return input_size;
}

// Predict function with forward pass through LSTM layers and fully connected layer
std::vector<float> NormalDataPredictor::predict(const std::vector<float>& input) {
    std::vector<float> output = input;
    std::vector<float> h, c;

    for (size_t i = 0; i < lstm_layers.size(); ++i) {
        // Initialize hidden and cell states for the current layer if needed
        int hidden_size = lstm_layers[i].get_hidden_size();
        h = std::vector<float>(hidden_size, 0.0f);
        c = std::vector<float>(hidden_size, 0.0f);

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
    if (fc_weight.empty() || fc_bias.empty()) {
        std::cerr << "Error: Fully connected layer weights or biases are empty." << std::endl;
        return {};
    }

    if (input.size() != fc_weight[0].size()) {
        std::cerr << "Error: Input size does not match fully connected layer input size (" 
                  << input.size() << " vs " << fc_weight[0].size() << ")." << std::endl;
        return {};
    }

    // Perform matrix-vector multiplication
    std::vector<float> result(fc_weight.size(), 0.0f);
    for (size_t i = 0; i < fc_weight.size(); ++i) {
        for (size_t j = 0; j < fc_weight[i].size(); ++j) {
            result[i] += fc_weight[i][j] * input[j];
        }
        result[i] += fc_bias[i];
        // Apply sigmoid activation
        result[i] = 1.0f / (1.0f + std::exp(-result[i]));
    }

    return result;
}

