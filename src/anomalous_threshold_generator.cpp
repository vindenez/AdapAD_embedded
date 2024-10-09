#include "anomalous_threshold_generator.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>  // For debug statements

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
                weight_ih_cell, weight_hh_cell, bias_ih_cell, bias_hh_cell) {}

// Constructor to initialize using hyperparameters
AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lookback_len, int prediction_len, float lower_bound, float upper_bound)
    : lookback_len(lookback_len), prediction_len(prediction_len), lower_bound(lower_bound), upper_bound(upper_bound),
      generator(lookback_len, prediction_len, 1, lookback_len) {}

// Update function for feed-forward adaptation of the generator
void AnomalousThresholdGenerator::update(int num_epochs, float learning_rate, const std::vector<float>& past_errors, float recent_error) {
    if (past_errors.empty()) {
        std::cerr << "Warning: past_errors vector is empty in update(). Cannot update generator." << std::endl;
        return;
    }

    std::vector<float> new_input = past_errors;
    new_input.push_back(recent_error);

    // Debugging: Print input size and content
    std::cout << "Updating generator with new_input size: " << new_input.size() << " Content: ";
    for (float val : new_input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Assuming one epoch update for incremental learning
    std::vector<float> output = generator.forward(new_input);

    // Debugging: Check output of the forward pass
    if (output.empty()) {
        std::cerr << "Warning: Generator forward pass in update() returned an empty vector." << std::endl;
    } else {
        std::cout << "Generator forward pass in update() completed successfully with output size: " << output.size() << std::endl;
    }
}

float AnomalousThresholdGenerator::generate(const std::vector<float>& prediction_errors, float minimal_threshold) {
    if (prediction_errors.empty()) {
        std::cerr << "Error: prediction_errors vector is empty in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }

    // Debugging: Print size and content of prediction_errors
    std::cout << "Generating threshold with prediction_errors size: " << prediction_errors.size() << " Content: ";
    for (float val : prediction_errors) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::vector<float> threshold_vec = generator.forward(prediction_errors);

    // Debugging: Check if the forward pass returned an empty threshold_vec
    if (threshold_vec.empty()) {
        std::cerr << "Error: threshold_vec is empty after generator forward pass in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }

    // Debugging: Print threshold value
    std::cout << "Generated threshold value: " << threshold_vec[0] << std::endl;

    float threshold = std::max(minimal_threshold, threshold_vec[0]);
    return threshold;
}
