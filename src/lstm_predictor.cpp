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
      num_layers(1),
      lookback_len(1) {
    
    if (weight_ih.empty() || weight_hh.empty() || bias_ih.empty() || bias_hh.empty()) {
        throw std::runtime_error("Empty weight or bias vectors in LSTMPredictor constructor");
    }

    weight_ih_input = weight_ih;
    weight_hh_input = weight_hh;
    bias_ih_input = bias_ih;
    bias_hh_input = bias_hh;

    // Initialize h and c
    h = std::vector<float>(hidden_size, 0.0f);
    c = std::vector<float>(hidden_size, 0.0f);
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

        if (weight_ih_input.empty() || weight_hh_input.empty()) {
            throw std::runtime_error("Empty weight matrices in LSTM forward pass");
        }

        // Combine all gates into one operation
        std::vector<float> combined_gate = matrix_vector_mul(weight_ih_input, input);
        std::vector<float> hidden_mul = matrix_vector_mul(weight_hh_input, prev_h);
        combined_gate = elementwise_add(combined_gate, hidden_mul);
        combined_gate = elementwise_add(combined_gate, bias_ih_input);
        combined_gate = elementwise_add(combined_gate, bias_hh_input);

        // Split the combined gate into individual gates
        std::vector<float> input_gate(hidden_size);
        std::vector<float> forget_gate(hidden_size);
        std::vector<float> cell_gate(hidden_size);
        std::vector<float> output_gate(hidden_size);

        for (int i = 0; i < hidden_size; ++i) {
            input_gate[i] = sigmoid(combined_gate[i]);
            forget_gate[i] = sigmoid(combined_gate[i + hidden_size]);
            cell_gate[i] = tanh_func(combined_gate[i + 2 * hidden_size]);
            output_gate[i] = sigmoid(combined_gate[i + 3 * hidden_size]);
        }

        // Update cell state
        std::vector<float> new_c(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            new_c[i] = forget_gate[i] * prev_c[i] + input_gate[i] * cell_gate[i];
        }

        // Calculate new hidden state
        std::vector<float> new_h(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            new_h[i] = output_gate[i] * tanh_func(new_c[i]);
        }

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