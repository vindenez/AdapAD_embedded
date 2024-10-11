#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Constructor to initialize with input size, hidden size, and other hyperparameters
LSTMPredictor::LSTMPredictor(int input_size, int hidden_size, int num_layers, int lookback_len)
    : input_size(input_size),
      hidden_size(hidden_size),
      num_layers(num_layers),
      lookback_len(lookback_len) {
    
    // Initialize weights and biases with Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    auto xavier_init = [&gen](int in_features, int out_features) {
        float std = std::sqrt(2.0f / (in_features + out_features));
        std::normal_distribution<> d(0, std);
        return [d = std::move(d), &gen]() mutable { return d(gen); };
    };

    auto init_weights = [&](int rows, int cols) {
        auto init = xavier_init(rows, cols);
        std::vector<std::vector<float>> w(rows, std::vector<float>(cols));
        for (auto& row : w) {
            for (auto& val : row) {
                val = init();
            }
        }
        return w;
    };

    auto init_bias = [](int size) {
        return std::vector<float>(size, 0.0f);  // Initialize biases to zero
    };

    // LSTM weights and biases for all gates
    weight_ih_input = init_weights(hidden_size, input_size);
    weight_hh_input = init_weights(hidden_size, hidden_size);
    bias_ih_input = init_bias(hidden_size);
    bias_hh_input = init_bias(hidden_size);

    weight_ih_forget = init_weights(hidden_size, input_size);
    weight_hh_forget = init_weights(hidden_size, hidden_size);
    bias_ih_forget = init_bias(hidden_size);
    bias_hh_forget = init_bias(hidden_size);

    weight_ih_cell = init_weights(hidden_size, input_size);
    weight_hh_cell = init_weights(hidden_size, hidden_size);
    bias_ih_cell = init_bias(hidden_size);
    bias_hh_cell = init_bias(hidden_size);

    weight_ih_output = init_weights(hidden_size, input_size);
    weight_hh_output = init_weights(hidden_size, hidden_size);
    bias_ih_output = init_bias(hidden_size);
    bias_hh_output = init_bias(hidden_size);

}

// Constructor to initialize with weights and biases for each gate
LSTMPredictor::LSTMPredictor(const std::vector<std::vector<float>>& weight_ih_input,
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
    : input_size(weight_ih_input[0].size()), hidden_size(weight_ih_input.size()),
      weight_ih_input(weight_ih_input), weight_hh_input(weight_hh_input), bias_ih_input(bias_ih_input), bias_hh_input(bias_hh_input),
      weight_ih_forget(weight_ih_forget), weight_hh_forget(weight_hh_forget), bias_ih_forget(bias_ih_forget), bias_hh_forget(bias_hh_forget),
      weight_ih_output(weight_ih_output), weight_hh_output(weight_hh_output), bias_ih_output(bias_ih_output), bias_hh_output(bias_hh_output),
      weight_ih_cell(weight_ih_cell), weight_hh_cell(weight_hh_cell), bias_ih_cell(bias_ih_cell), bias_hh_cell(bias_hh_cell),
      h(hidden_size, 0.0f), c(hidden_size, 0.0f) {
    
}

// Simplified constructor for fewer parameters (matching NormalDataPredictor)
LSTMPredictor::LSTMPredictor(const std::vector<std::vector<float>>& weight_ih,
                             const std::vector<std::vector<float>>& weight_hh,
                             const std::vector<float>& bias_ih,
                             const std::vector<float>& bias_hh,
                             int input_size,
                             int hidden_size)
    : input_size(input_size),
      hidden_size(hidden_size),
      num_layers(1),  // Assuming single layer for simplified constructor
      lookback_len(1),  // Assuming lookback of 1 for simplified constructor
      weight_ih_input(weight_ih.begin(), weight_ih.begin() + hidden_size),
      weight_hh_input(weight_hh.begin(), weight_hh.begin() + hidden_size),
      bias_ih_input(bias_ih.begin(), bias_ih.begin() + hidden_size),
      bias_hh_input(bias_hh.begin(), bias_hh.begin() + hidden_size),
      weight_ih_forget(weight_ih.begin() + hidden_size, weight_ih.begin() + 2 * hidden_size),
      weight_hh_forget(weight_hh.begin() + hidden_size, weight_hh.begin() + 2 * hidden_size),
      bias_ih_forget(bias_ih.begin() + hidden_size, bias_ih.begin() + 2 * hidden_size),
      bias_hh_forget(bias_hh.begin() + hidden_size, bias_hh.begin() + 2 * hidden_size),
      weight_ih_output(weight_ih.begin() + 3 * hidden_size, weight_ih.end()),
      weight_hh_output(weight_hh.begin() + 3 * hidden_size, weight_hh.end()),
      bias_ih_output(bias_ih.begin() + 3 * hidden_size, bias_ih.end()),
      bias_hh_output(bias_hh.begin() + 3 * hidden_size, bias_hh.end()),
      weight_ih_cell(weight_ih.begin() + 2 * hidden_size, weight_ih.begin() + 3 * hidden_size),
      weight_hh_cell(weight_hh.begin() + 2 * hidden_size, weight_hh.begin() + 3 * hidden_size),
      bias_ih_cell(bias_ih.begin() + 2 * hidden_size, bias_ih.begin() + 3 * hidden_size),
      bias_hh_cell(bias_hh.begin() + 2 * hidden_size, bias_hh.begin() + 3 * hidden_size),
      h(hidden_size, 0.0f),
      c(hidden_size, 0.0f) {

    if (weight_ih.empty() || (weight_ih.size() > 0 && weight_ih[0].empty()) ||
        weight_hh.empty() || (weight_hh.size() > 0 && weight_hh[0].empty()) ||
        bias_ih.empty() || bias_hh.empty()) {
        throw std::runtime_error("LSTM initialization error: Empty weights or biases");
    }

    if (weight_ih.size() != weight_hh.size() || weight_ih.size() != bias_ih.size() || weight_ih.size() != bias_hh.size()) {
        throw std::runtime_error("LSTM initialization error: Inconsistent dimensions");
    }

    std::cout << "LSTMPredictor initialized with:" << std::endl;
    std::cout << "input_size: " << input_size << std::endl;
    std::cout << "hidden_size: " << hidden_size << std::endl;
}

// Copy constructor implementation
LSTMPredictor::LSTMPredictor(const LSTMPredictor& other)
    : input_size(other.input_size), hidden_size(other.hidden_size),
      num_layers(other.num_layers), lookback_len(other.lookback_len),
      weight_ih_input(other.weight_ih_input), weight_hh_input(other.weight_hh_input),
      bias_ih_input(other.bias_ih_input), bias_hh_input(other.bias_hh_input),
      weight_ih_forget(other.weight_ih_forget), weight_hh_forget(other.weight_hh_forget),
      bias_ih_forget(other.bias_ih_forget), bias_hh_forget(other.bias_hh_forget),
      weight_ih_output(other.weight_ih_output), weight_hh_output(other.weight_hh_output),
      bias_ih_output(other.bias_ih_output), bias_hh_output(other.bias_hh_output),
      weight_ih_cell(other.weight_ih_cell), weight_hh_cell(other.weight_hh_cell),
      bias_ih_cell(other.bias_ih_cell), bias_hh_cell(other.bias_hh_cell),
      h(other.h), c(other.c) {
}

// Method to get input size
int LSTMPredictor::get_input_size() const {
    return input_size;
}

// Method to get hidden size
int LSTMPredictor::get_hidden_size() const {
    return hidden_size;
}

// Forward pass through the LSTM layer
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> LSTMPredictor::forward(
    const std::vector<float>& input,
    const std::vector<float>& prev_h,
    const std::vector<float>& prev_c) {
    try {
        if (input.size() != input_size) {
            throw std::runtime_error("Input size mismatch. Expected: " + std::to_string(input_size) + ", Got: " + std::to_string(input.size()));
        }

        // Input gate
        std::vector<float> input_gate = matrix_vector_mul(weight_ih_input, input);

        std::vector<float> hidden_mul = matrix_vector_mul(weight_hh_input, prev_h);

        input_gate = elementwise_add(input_gate, hidden_mul);

        input_gate = elementwise_add(input_gate, bias_ih_input);
        input_gate = elementwise_add(input_gate, bias_hh_input);
        for (float& val : input_gate) {
            val = sigmoid(val);
        }

        // Forget gate
        if (weight_ih_forget.empty() || weight_hh_forget.empty() || bias_ih_forget.empty() || bias_hh_forget.empty()) {
            throw std::runtime_error("Forget gate weights or biases are empty");
        }

        std::vector<float> forget_gate = matrix_vector_mul(weight_ih_forget, input);
        std::vector<float> hidden_mul_forget = matrix_vector_mul(weight_hh_forget, prev_h);
        forget_gate = elementwise_add(forget_gate, hidden_mul_forget);
        forget_gate = elementwise_add(forget_gate, bias_ih_forget);
        forget_gate = elementwise_add(forget_gate, bias_hh_forget);
        for (float& val : forget_gate) {
            val = sigmoid(val);
        }

        // Output gate
        std::vector<float> output_gate = matrix_vector_mul(weight_ih_output, input);
        hidden_mul = matrix_vector_mul(weight_hh_output, prev_h);
        output_gate = elementwise_add(output_gate, hidden_mul);
        output_gate = elementwise_add(output_gate, bias_ih_output);
        output_gate = elementwise_add(output_gate, bias_hh_output);
        for (float& val : output_gate) {
            val = sigmoid(val);
        }

        // Cell state update
        std::vector<float> g = matrix_vector_mul(weight_ih_cell, input);
        hidden_mul = matrix_vector_mul(weight_hh_cell, prev_h);
        g = elementwise_add(g, hidden_mul);
        g = elementwise_add(g, bias_ih_cell);
        g = elementwise_add(g, bias_hh_cell);
        for (float& val : g) {
            val = tanh_func(val);
        }

        // Update cell state c
        std::vector<float> new_c = elementwise_mul(forget_gate, prev_c);
        new_c = elementwise_add(new_c, elementwise_mul(input_gate, g));

        // Calculate new hidden state h
        std::vector<float> tanh_c(new_c.size());
        for (size_t j = 0; j < new_c.size(); ++j) {
            tanh_c[j] = tanh_func(new_c[j]);
        }
        std::vector<float> new_h = elementwise_mul(output_gate, tanh_c);

        return {new_h, new_h, new_c};
    } catch (const std::exception& e) {
        std::cerr << "Error in LSTM forward pass: " << e.what() << std::endl;
        return {{}, {}, {}};
    }
}

void LSTMPredictor::update_parameters(const std::vector<std::vector<float>>& dw_ih, 
                                      const std::vector<std::vector<float>>& dw_hh,
                                      const std::vector<float>& db_ih, 
                                      const std::vector<float>& db_hh, 
                                      float learning_rate) {
    
    // Update input-hidden weights
    for (size_t i = 0; i < weight_ih_input.size(); ++i) {
        for (size_t j = 0; j < weight_ih_input[i].size(); ++j) {
            weight_ih_input[i][j] -= learning_rate * dw_ih[i][j];
        }
    }

    // Update hidden-hidden weights
    for (size_t i = 0; i < weight_hh_input.size(); ++i) {
        for (size_t j = 0; j < weight_hh_input[i].size(); ++j) {
            weight_hh_input[i][j] -= learning_rate * dw_hh[i][j];
        }
    }

    // Update input-hidden biases
    for (size_t i = 0; i < bias_ih_input.size(); ++i) {
        bias_ih_input[i] -= learning_rate * db_ih[i];
    }

    // Update hidden-hidden biases
    for (size_t i = 0; i < bias_hh_input.size(); ++i) {
        bias_hh_input[i] -= learning_rate * db_hh[i];
    }

}