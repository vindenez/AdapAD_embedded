#include "anomalous_threshold_generator.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Constructor to initialize using weights and biases for each gate
AnomalousThresholdGenerator::AnomalousThresholdGenerator(
    const std::vector<std::vector<float>>& weight_ih_input,
    const std::vector<std::vector<float>>& weight_hh_input,
    const std::vector<float>& bias_ih_input,
    const std::vector<float>& bias_hh_input,
    const std::vector<std::vector<float>>& weight_ih_forget,
    const std::vector<std::vector<float>>& weight_hh_forget,
    const std::vector<float>& bias_ih_forget,
    const std::vector<float>& bias_hh_forget,
    const std::vector<std::vector<float>>& weight_ih_output,
    const std::vector<std::vector<float>>& weight_hh_output,
    const std::vector<float>& bias_ih_output,
    const std::vector<float>& bias_hh_output,
    const std::vector<std::vector<float>>& weight_ih_cell,
    const std::vector<std::vector<float>>& weight_hh_cell,
    const std::vector<float>& bias_ih_cell,
    const std::vector<float>& bias_hh_cell)
    : generator(weight_ih_input, weight_hh_input, bias_ih_input, bias_hh_input,
                weight_ih_forget, weight_hh_forget, bias_ih_forget, bias_hh_forget,
                weight_ih_output, weight_hh_output, bias_ih_output, bias_hh_output,
                weight_ih_cell, weight_hh_cell, bias_ih_cell, bias_hh_cell),
      h(generator.get_hidden_size(), 0.0f),
      c(generator.get_hidden_size(), 0.0f) {
    
    // Initialize other members
    lookback_len = weight_ih_input[0].size();
    prediction_len = 1; // Assuming single step prediction
    lower_bound = 0.0f; // Set default values or add parameters to constructor
    upper_bound = 1.0f; // Set default values or add parameters to constructor
}

// Constructor to initialize using hyperparameters
AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lookback_len, int prediction_len, float lower_bound, float upper_bound)
    : lookback_len(lookback_len), prediction_len(prediction_len), lower_bound(lower_bound), upper_bound(upper_bound),
      generator(lookback_len, prediction_len, 1, lookback_len),
      h(generator.get_hidden_size(), 0.0f),
      c(generator.get_hidden_size(), 0.0f) {}

// Update function for feed-forward adaptation of the generator
void AnomalousThresholdGenerator::update(int num_epochs, float learning_rate, const std::vector<float>& past_errors, float recent_error) {
    if (past_errors.empty()) {
        std::cerr << "Warning: past_errors vector is empty in update(). Cannot update generator." << std::endl;
        return;
    }

    // Create new_input by taking the last 2 past errors and the recent error
    std::vector<float> new_input;
    if (past_errors.size() >= 2) {
        new_input.push_back(past_errors[past_errors.size() - 2]);
        new_input.push_back(past_errors[past_errors.size() - 1]);
    } else if (past_errors.size() == 1) {
        new_input.push_back(past_errors[0]);
        new_input.push_back(0.0f); // Add a placeholder value
    } else {
        new_input.push_back(0.0f); // Add two placeholder values
        new_input.push_back(0.0f);
    }
    new_input.push_back(recent_error);

    std::cout << "new_input size: " << new_input.size() << std::endl;

    // Forward pass and backpropagation
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto [output, new_h, new_c] = generator.forward(new_input, h, c);
        
    }

}

std::tuple<std::vector<float>, std::vector<float>, 
           std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<float>, std::vector<float>>
AnomalousThresholdGenerator::backward_step(const std::vector<float>& input,
                                           const std::vector<float>& h_prev,
                                           const std::vector<float>& c_prev,
                                           const std::vector<float>& doutput) {
    // Get the current weights and biases
    auto weight_ih = generator.get_weight_ih_input();
    auto weight_hh = generator.get_weight_hh_input();
    auto bias_ih = generator.get_bias_ih_input();
    auto bias_hh = generator.get_bias_hh_input();

    // Initialize gradients
    std::vector<std::vector<float>> dw_ih(weight_ih.size(), std::vector<float>(weight_ih[0].size(), 0.0f));
    std::vector<std::vector<float>> dw_hh(weight_hh.size(), std::vector<float>(weight_hh[0].size(), 0.0f));
    std::vector<float> db_ih(bias_ih.size(), 0.0f);
    std::vector<float> db_hh(bias_hh.size(), 0.0f);

    // Compute gradients for the output gate
    std::vector<float> dh = doutput;
    std::vector<float> dc(c_prev.size(), 0.0f);

    // Compute gradients for the input gate
    for (size_t i = 0; i < dh.size(); ++i) {
        float dh_i = dh[i] * tanh_func(c_prev[i]);
        dc[i] += dh[i] * h_prev[i] * (1 - tanh_func(c_prev[i]) * tanh_func(c_prev[i]));
        
        for (size_t j = 0; j < input.size(); ++j) {
            dw_ih[i][j] += dh_i * input[j];
        }
        for (size_t j = 0; j < h_prev.size(); ++j) {
            dw_hh[i][j] += dh_i * h_prev[j];
        }
        db_ih[i] += dh_i;
        db_hh[i] += dh_i;
    }

    // Compute gradients for the forget gate
    for (size_t i = 0; i < dc.size(); ++i) {
        float dc_i = dc[i] * c_prev[i];
        for (size_t j = 0; j < input.size(); ++j) {
            dw_ih[i + dh.size()][j] += dc_i * input[j];
        }
        for (size_t j = 0; j < h_prev.size(); ++j) {
            dw_hh[i + dh.size()][j] += dc_i * h_prev[j];
        }
        db_ih[i + dh.size()] += dc_i;
        db_hh[i + dh.size()] += dc_i;
    }

    return {dh, dc, dw_ih, dw_hh, db_ih, db_hh};
}

void AnomalousThresholdGenerator::update_parameters(const std::vector<std::vector<float>>& dw_ih,
                                                    const std::vector<std::vector<float>>& dw_hh,
                                                    const std::vector<float>& db_ih,
                                                    const std::vector<float>& db_hh,
                                                    float learning_rate) {
    auto weight_ih = generator.get_weight_ih_input();
    auto weight_hh = generator.get_weight_hh_input();
    auto bias_ih = generator.get_bias_ih_input();
    auto bias_hh = generator.get_bias_hh_input();

    // Update weights and biases
    for (size_t i = 0; i < weight_ih.size(); ++i) {
        for (size_t j = 0; j < weight_ih[i].size(); ++j) {
            weight_ih[i][j] -= learning_rate * dw_ih[i][j];
        }
    }

    for (size_t i = 0; i < weight_hh.size(); ++i) {
        for (size_t j = 0; j < weight_hh[i].size(); ++j) {
            weight_hh[i][j] -= learning_rate * dw_hh[i][j];
        }
    }

    for (size_t i = 0; i < bias_ih.size(); ++i) {
        bias_ih[i] -= learning_rate * db_ih[i];
    }

    for (size_t i = 0; i < bias_hh.size(); ++i) {
        bias_hh[i] -= learning_rate * db_hh[i];
    }

    // Set updated weights and biases
    generator.set_weight_ih_input(weight_ih);
    generator.set_weight_hh_input(weight_hh);
    generator.set_bias_ih_input(bias_ih);
    generator.set_bias_hh_input(bias_hh);
}

float AnomalousThresholdGenerator::generate(const std::vector<float>& prediction_errors, float minimal_threshold) {
    if (prediction_errors.empty()) {
        std::cerr << "Error: prediction_errors vector is empty in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }
 
    std::tie(output, h, c) = generator.forward(prediction_errors, h, c);
 
    if (output.empty()) {
        std::cerr << "Error: output is empty after generator forward pass in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }
 
    float threshold = std::max(minimal_threshold, output[0]);
    std::cout << "Generated threshold: " << threshold << std::endl;
 
    return threshold;
}

float AnomalousThresholdGenerator::generate_threshold(const std::vector<float>& new_input) {
    if (new_input.size() != lookback_len) {
        throw std::invalid_argument("Input size does not match lookback length");
    }

    // Forward pass through the LSTM
    std::tie(output, h, c) = generator.forward(new_input, h, c);

    // Calculate the mean of the output
    float mean = std::accumulate(output.begin(), output.end(), 0.0f) / output.size();

    // Clamp the mean between lower_bound and upper_bound
    return std::clamp(mean, lower_bound, upper_bound);
}

std::vector<float> AnomalousThresholdGenerator::generate_thresholds(const std::vector<std::vector<float>>& input_sequence) {
    std::vector<float> thresholds;
    thresholds.reserve(input_sequence.size());

    for (const auto& input : input_sequence) {
        thresholds.push_back(generate_threshold(input));
    }

    return thresholds;
}