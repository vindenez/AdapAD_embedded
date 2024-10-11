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
void AnomalousThresholdGenerator::update(int num_epochs, float learning_rate, const std::vector<float>& past_errors) {
    std::cout << "Entering update function" << std::endl;
    if (past_errors.size() != lookback_len) {
        std::cerr << "Error: Invalid input size in generator update. Expected: " << lookback_len 
                  << ", Got: " << past_errors.size() << std::endl;
        return;
    }

    std::cout << "Updating generator with input: ";
    for (float val : past_errors) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Forward pass and backpropagation
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << " of " << num_epochs << std::endl;
        try {
            auto [output, new_h, new_c] = generator.forward(past_errors, h, c);
            
            if (output.empty()) {
                std::cerr << "Error: Empty output from generator forward pass" << std::endl;
                return;
            }

            // Compute loss (for example, mean squared error)
            float loss = 0.0f;
            for (float val : output) {
                loss += (val - past_errors.back()) * (val - past_errors.back());
            }
            loss /= output.size();

            // Compute gradient of loss with respect to output
            std::vector<float> doutput(output.size());
            for (size_t i = 0; i < output.size(); ++i) {
                doutput[i] = 2.0f * (output[i] - past_errors.back()) / output.size();
            }

            std::cout << "Starting backward pass" << std::endl;
            auto [dh, dc, dw_ih, dw_hh, db_ih, db_hh] = backward_step(past_errors, h, c, doutput);

            std::cout << "Updating parameters" << std::endl;
            update_parameters(dw_ih, dw_hh, db_ih, db_hh, learning_rate);

            // Update hidden state and cell state
            h = new_h;
            c = new_c;

            std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in epoch " << epoch + 1 << ": " << e.what() << std::endl;
            return;
        }
    }
    std::cout << "Update function completed" << std::endl;
}

std::tuple<std::vector<float>, std::vector<float>, 
           std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<float>, std::vector<float>>
AnomalousThresholdGenerator::backward_step(const std::vector<float>& input,
                                           const std::vector<float>& h_prev,
                                           const std::vector<float>& c_prev,
                                           const std::vector<float>& doutput) {
    std::cout << "Entering backward_step" << std::endl;

    // Get the current weights and biases
    auto weight_ih = generator.get_weight_ih_input();
    auto weight_hh = generator.get_weight_hh_input();
    auto bias_ih = generator.get_bias_ih_input();
    auto bias_hh = generator.get_bias_hh_input();

    std::cout << "Weights and biases retrieved" << std::endl;

    // Check dimensions and print sizes
    std::cout << "Dimensions: " << std::endl;
    std::cout << "weight_ih: " << weight_ih.size() << "x" << (weight_ih.empty() ? 0 : weight_ih[0].size()) << std::endl;
    std::cout << "weight_hh: " << weight_hh.size() << "x" << (weight_hh.empty() ? 0 : weight_hh[0].size()) << std::endl;
    std::cout << "bias_ih: " << bias_ih.size() << std::endl;
    std::cout << "bias_hh: " << bias_hh.size() << std::endl;
    std::cout << "input: " << input.size() << std::endl;
    std::cout << "h_prev: " << h_prev.size() << std::endl;
    std::cout << "c_prev: " << c_prev.size() << std::endl;
    std::cout << "doutput: " << doutput.size() << std::endl;

    if (weight_ih.empty() || weight_ih[0].empty() || weight_hh.empty() || weight_hh[0].empty() ||
        bias_ih.empty() || bias_hh.empty() || input.empty() || h_prev.empty() || c_prev.empty() || doutput.empty()) {
        throw std::runtime_error("One or more input vectors/matrices are empty in backward_step");
    }

    if (weight_ih.size() != weight_hh.size() || weight_ih.size() != bias_ih.size() || weight_ih.size() != bias_hh.size()) {
        throw std::runtime_error("Inconsistent sizes in weight and bias matrices/vectors");
    }

    if (input.size() != weight_ih[0].size() || h_prev.size() != weight_hh[0].size() || 
        c_prev.size() != h_prev.size() || doutput.size() != h_prev.size()) {
        throw std::runtime_error("Inconsistent input sizes in backward_step");
    }

    // Initialize gradients
    std::vector<std::vector<float>> dw_ih(weight_ih.size(), std::vector<float>(weight_ih[0].size(), 0.0f));
    std::vector<std::vector<float>> dw_hh(weight_hh.size(), std::vector<float>(weight_hh[0].size(), 0.0f));
    std::vector<float> db_ih(bias_ih.size(), 0.0f);
    std::vector<float> db_hh(bias_hh.size(), 0.0f);

    std::cout << "Gradients initialized" << std::endl;

    // Compute gradients for the output gate
    std::vector<float> dh = doutput;
    std::vector<float> dc(c_prev.size(), 0.0f);

    std::cout << "Starting gradient computation" << std::endl;

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

    std::cout << "Input gate gradients computed" << std::endl;

    // Compute gradients for the forget gate
    for (size_t i = 0; i < dc.size(); ++i) {
        float dc_i = dc[i] * c_prev[i];
        for (size_t j = 0; j < input.size(); ++j) {
            if (i >= dw_ih.size() || j >= dw_ih[i].size()) {
                std::cerr << "Index out of bounds: i=" << i << ", j=" << j << ", dw_ih size=" << dw_ih.size() << "x" << (dw_ih.empty() ? 0 : dw_ih[0].size()) << std::endl;
                throw std::runtime_error("Index out of bounds in forget gate gradient computation (dw_ih)");
            }
            dw_ih[i][j] += dc_i * input[j];
        }
        for (size_t j = 0; j < h_prev.size(); ++j) {
            if (i >= dw_hh.size() || j >= dw_hh[i].size()) {
                std::cerr << "Index out of bounds: i=" << i << ", j=" << j << ", dw_hh size=" << dw_hh.size() << "x" << (dw_hh.empty() ? 0 : dw_hh[0].size()) << std::endl;
                throw std::runtime_error("Index out of bounds in forget gate gradient computation (dw_hh)");
            }
            dw_hh[i][j] += dc_i * h_prev[j];
        }
        if (i >= db_ih.size() || i >= db_hh.size()) {
            std::cerr << "Index out of bounds: i=" << i << ", db_ih size=" << db_ih.size() << ", db_hh size=" << db_hh.size() << std::endl;
            throw std::runtime_error("Index out of bounds in forget gate gradient computation (db_ih/db_hh)");
        }
        db_ih[i] += dc_i;
        db_hh[i] += dc_i;
    }

    std::cout << "Forget gate gradients computed" << std::endl;

    std::cout << "Backward step completed" << std::endl;
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
    if (prediction_errors.empty() || prediction_errors.size() != lookback_len) {
        std::cerr << "Error: Invalid prediction_errors size in generate(). Expected: " << lookback_len 
                  << ", Got: " << prediction_errors.size() << std::endl;
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