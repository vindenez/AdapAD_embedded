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
            throw std::runtime_error("Missing required LSTM layer weights/biases for layer " + std::to_string(i));
        }

        // Debugging weights and biases for each LSTM layer
        std::cout << "Loading LSTM Layer " << i << "..." << std::endl;
        std::cout << "Weight_ih dimensions: " << weights.at(weight_ih_key).size() << " x " 
                  << (weights.at(weight_ih_key).empty() ? 0 : weights.at(weight_ih_key)[0].size()) << std::endl;
        std::cout << "Weight_hh dimensions: " << weights.at(weight_hh_key).size() << " x " 
                  << (weights.at(weight_hh_key).empty() ? 0 : weights.at(weight_hh_key)[0].size()) << std::endl;
        std::cout << "Bias_ih size: " << biases.at(bias_ih_key).size() << std::endl;
        std::cout << "Bias_hh size: " << biases.at(bias_hh_key).size() << std::endl;

        // Create LSTM layer
        lstm_layers.emplace_back(weights.at(weight_ih_key), weights.at(weight_hh_key),
                                 biases.at(bias_ih_key), biases.at(bias_hh_key));
    }

    // Load fully connected layer
    if (weights.find("fc.weight") != weights.end() && biases.find("fc.bias") != biases.end()) {
        fc_weight = weights.at("fc.weight");
        fc_bias = biases.at("fc.bias");

        std::cout << "Fully Connected Layer Weight dimensions: " << fc_weight.size() << " x " 
                  << (fc_weight.empty() ? 0 : fc_weight[0].size()) << std::endl;
        std::cout << "Fully Connected Layer Bias size: " << fc_bias.size() << std::endl;
    } else {
        throw std::runtime_error("Missing fully connected layer weights/biases");
    }

    // Set input size based on the first LSTM layer
    input_size = weights.at("lstm.weight_ih_l0")[0].size();

    std::cout << "NormalDataPredictor initialized with:" << std::endl;
    std::cout << "Input size: " << input_size << std::endl;
    std::cout << "Number of LSTM layers: " << lstm_layers.size() << std::endl;
    for (size_t i = 0; i < lstm_layers.size(); ++i) {
        std::cout << "LSTM layer " << i << " hidden size: " << lstm_layers[i].get_hidden_size() << std::endl;
    }
    std::cout << "Fully connected layer input size: " << fc_weight[0].size() << std::endl;
    std::cout << "Fully connected layer output size: " << fc_weight.size() << std::endl;
}


// Method to get input size
int NormalDataPredictor::get_input_size() const {
    return input_size;
}

// Predict function with forward pass through LSTM layers and fully connected layer
std::vector<float> NormalDataPredictor::predict(const std::vector<float>& input) {
    std::cout << "Predict function called with input size: " << input.size() << std::endl;
    std::cout << "Expected input size: " << input_size << std::endl;

    if (input.size() != input_size) {
        throw std::runtime_error("Input size mismatch in predict function. Expected: " + 
                                 std::to_string(input_size) + ", Got: " + std::to_string(input.size()));
    }

    std::vector<float> output = input;
    std::vector<float> h(lstm_layers[0].get_hidden_size(), 0.0f);
    std::vector<float> c(lstm_layers[0].get_hidden_size(), 0.0f);

    for (size_t i = 0; i < lstm_layers.size(); ++i) {
        std::cout << "LSTM layer " << i << " input size: " << output.size() << std::endl;
        std::cout << "LSTM layer " << i << " hidden size: " << lstm_layers[i].get_hidden_size() << std::endl;
        std::tie(output, h, c) = lstm_layers[i].forward(output, h, c);
        if (output.empty()) {
            throw std::runtime_error("Error: LSTM layer " + std::to_string(i) + " forward pass returned an empty vector.");
        }
        std::cout << "LSTM layer " << i << " output size: " << output.size() << std::endl;
    }

    output = fully_connected(output);
    std::cout << "Fully connected layer output size: " << output.size() << std::endl;

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

void NormalDataPredictor::update(int num_epochs, float learning_rate, const std::vector<float>& past_observations, float recent_observation) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::vector<float> h(lstm_layers[0].get_hidden_size(), 0.0f);
        std::vector<float> c(lstm_layers[0].get_hidden_size(), 0.0f);
        std::vector<float> output = past_observations;

        // Forward pass
        for (auto& lstm : lstm_layers) {
            std::tie(output, h, c) = lstm.forward(output, h, c);
        }
        output = fully_connected(output);

        // Compute loss
        float loss = std::pow(output[0] - recent_observation, 2);  // MSE loss

        // Backward pass (simplified)
        float d_output = 2 * (output[0] - recent_observation);
        std::vector<float> d_fc(fc_weight[0].size(), 0.0f);
        for (size_t i = 0; i < fc_weight.size(); ++i) {
            for (size_t j = 0; j < fc_weight[i].size(); ++j) {
                d_fc[j] += d_output * fc_weight[i][j];
            }
        }

        // Update fully connected layer
        for (size_t i = 0; i < fc_weight.size(); ++i) {
            for (size_t j = 0; j < fc_weight[i].size(); ++j) {
                fc_weight[i][j] -= learning_rate * d_output * output[j];
            }
            fc_bias[i] -= learning_rate * d_output;
        }

        // Backward through LSTM layers (simplified)
        std::vector<float> d_h = d_fc;
        for (auto it = lstm_layers.rbegin(); it != lstm_layers.rend(); ++it) {
            auto& lstm = *it;
            std::vector<std::vector<float>> dw_ih(lstm.get_weight_ih_input().size(), std::vector<float>(lstm.get_weight_ih_input()[0].size(), 0.0f));
            std::vector<std::vector<float>> dw_hh(lstm.get_weight_hh_input().size(), std::vector<float>(lstm.get_weight_hh_input()[0].size(), 0.0f));
            std::vector<float> db_ih(lstm.get_bias_ih_input().size(), 0.0f);
            std::vector<float> db_hh(lstm.get_bias_hh_input().size(), 0.0f);

            // Compute gradients (simplified)
            for (size_t i = 0; i < d_h.size(); ++i) {
                for (size_t j = 0; j < past_observations.size(); ++j) {
                    dw_ih[i][j] = d_h[i] * past_observations[j];
                }
                for (size_t j = 0; j < h.size(); ++j) {
                    dw_hh[i][j] = d_h[i] * h[j];
                }
                db_ih[i] = d_h[i];
                db_hh[i] = d_h[i];
            }

            // Update LSTM parameters
            lstm.update_parameters(dw_ih, dw_hh, db_ih, db_hh, learning_rate);

            // Propagate error (simplified)
            d_h = std::vector<float>(h.size(), 0.0f);
            for (size_t i = 0; i < d_h.size(); ++i) {
                for (size_t j = 0; j < past_observations.size(); ++j) {
                    d_h[i] += dw_ih[i][j];
                }
            }
        }

        // Early stopping condition (if needed)
        // if (loss < some_threshold) break;
    }
}