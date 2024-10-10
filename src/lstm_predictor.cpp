#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <iostream>

// Constructor to initialize with input size, hidden size, and other hyperparameters
LSTMPredictor::LSTMPredictor(int input_size, int hidden_size, int num_layers, int lookback_len)
    : input_size(input_size),
      hidden_size(hidden_size),
      num_layers(num_layers),
      lookback_len(lookback_len),
      h(hidden_size, 0.0f),
      c(hidden_size, 0.0f) {
    
    // Initialize weights and biases with random values
    weight_ih_input = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(input_size));
    weight_hh_input = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size));
    bias_ih_input = std::vector<float>(4 * hidden_size);
    bias_hh_input = std::vector<float>(4 * hidden_size);

    // Initialize with random values (you can use a proper random number generator)
    for (auto& row : weight_ih_input) {
        for (auto& val : row) {
            val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    for (auto& row : weight_hh_input) {
        for (auto& val : row) {
            val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    for (auto& val : bias_ih_input) {
        val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (auto& val : bias_hh_input) {
        val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    std::cout << "LSTMPredictor initialized with:" << std::endl;
    std::cout << "input_size: " << input_size << std::endl;
    std::cout << "hidden_size: " << hidden_size << std::endl;
    std::cout << "weight_ih dimensions: " << weight_ih_input.size() << " x " << weight_ih_input[0].size() << std::endl;
    std::cout << "weight_hh dimensions: " << weight_hh_input.size() << " x " << weight_hh_input[0].size() << std::endl;
    std::cout << "bias_ih size: " << bias_ih_input.size() << std::endl;
    std::cout << "bias_hh size: " << bias_hh_input.size() << std::endl;
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
      h(hidden_size, 0.0f), c(hidden_size, 0.0f) {}

// Simplified constructor for fewer parameters (matching NormalDataPredictor)
LSTMPredictor::LSTMPredictor(const std::vector<std::vector<float>>& weight_ih,
                             const std::vector<std::vector<float>>& weight_hh,
                             const std::vector<float>& bias_ih,
                             const std::vector<float>& bias_hh)
    : input_size(weight_ih.empty() ? 0 : weight_ih[0].size()),
      hidden_size(weight_ih.size()),
      weight_ih_input(weight_ih), weight_hh_input(weight_hh),
      bias_ih_input(bias_ih), bias_hh_input(bias_hh),
      h(hidden_size, 0.0f), c(hidden_size, 0.0f) {
    
    std::cout << "LSTMPredictor constructor called with:" << std::endl;
    std::cout << "weight_ih dimensions: " << weight_ih.size() << " x " << (weight_ih.empty() ? 0 : weight_ih[0].size()) << std::endl;
    std::cout << "weight_hh dimensions: " << weight_hh.size() << " x " << (weight_hh.empty() ? 0 : weight_hh[0].size()) << std::endl;
    std::cout << "bias_ih size: " << bias_ih.size() << std::endl;
    std::cout << "bias_hh size: " << bias_hh.size() << std::endl;

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
      h(other.h), c(other.c) {}

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
        // Check if input size matches expected input size
        if (input.size() != input_size) {
            std::cerr << "Error: Input size does not match expected input size (" << input.size() << " vs " << input_size << ")." << std::endl;
            return {{}, {}, {}};
        }

        // Check if hidden state size matches expected hidden size
        if (prev_h.size() != hidden_size) {
            std::cerr << "Error: Hidden state size does not match expected size (" << prev_h.size() << " vs " << hidden_size << ")." << std::endl;
            return {{}, {}, {}};
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
        std::vector<float> forget_gate = matrix_vector_mul(weight_ih_forget, input);
        hidden_mul = matrix_vector_mul(weight_hh_forget, prev_h);
        forget_gate = elementwise_add(forget_gate, hidden_mul);
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

        // Cell state update (intermediate state)
        std::vector<float> g = matrix_vector_mul(weight_ih_cell, input);
        hidden_mul = matrix_vector_mul(weight_hh_cell, prev_h);
        g = elementwise_add(g, hidden_mul);
        g = elementwise_add(g, bias_ih_cell);
        g = elementwise_add(g, bias_hh_cell);
        for (float& val : g) {
            val = tanh_func(val);
        }

        // Update cell state c
        std::vector<float> new_c = elementwise_add(elementwise_mul(forget_gate, prev_c), elementwise_mul(input_gate, g));

        // Calculate new hidden state h
        std::vector<float> tanh_c(new_c.size());
        for (size_t j = 0; j < new_c.size(); ++j) {
            tanh_c[j] = tanh_func(new_c[j]);
        }
        std::vector<float> new_h = elementwise_mul(output_gate, tanh_c);

        // Return updated hidden state, new hidden state, and new cell state
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