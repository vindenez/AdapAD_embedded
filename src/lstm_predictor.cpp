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
      c(hidden_size, 0.0f) {}

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
    : input_size(weight_ih[0].size()), hidden_size(weight_ih.size()),
      weight_ih_input(weight_ih), weight_hh_input(weight_hh),
      bias_ih_input(bias_ih), bias_hh_input(bias_hh),
      h(hidden_size, 0.0f), c(hidden_size, 0.0f) {}

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

// Forward pass through the LSTM layer
std::vector<float> LSTMPredictor::forward(const std::vector<float>& input) {
    // Check if input size matches expected input size
    if (input.size() != input_size) {
        std::cerr << "Error: Input size does not match expected input size (" << input.size() << " vs " << input_size << ")." << std::endl;
        return {};
    }

    // Input gate
    std::vector<float> input_gate = matrix_vector_mul(weight_ih_input, input);
    if (input_gate.empty()) return {}; // Return if multiplication failed

    std::vector<float> hidden_mul = matrix_vector_mul(weight_hh_input, h);
    if (hidden_mul.empty()) return {}; // Return if multiplication failed

    input_gate = elementwise_add(input_gate, hidden_mul);
    if (input_gate.size() != bias_ih_input.size()) {
        std::cerr << "Error: Dimension mismatch after hidden weight multiplication in input gate (expected " 
                  << bias_ih_input.size() << " but got " << input_gate.size() << ")." << std::endl;
        return {};
    }
    input_gate = elementwise_add(input_gate, bias_ih_input);
    for (float& val : input_gate) {
        val = sigmoid(val);
    }

    // Forget gate
    std::vector<float> forget_gate = matrix_vector_mul(weight_ih_forget, input);
    if (forget_gate.empty()) return {}; // Return if multiplication failed

    hidden_mul = matrix_vector_mul(weight_hh_forget, h);
    if (hidden_mul.empty()) return {}; // Return if multiplication failed

    forget_gate = elementwise_add(forget_gate, hidden_mul);
    if (forget_gate.size() != bias_ih_forget.size()) {
        std::cerr << "Error: Dimension mismatch after hidden weight multiplication in forget gate (expected " 
                  << bias_ih_forget.size() << " but got " << forget_gate.size() << ")." << std::endl;
        return {};
    }
    forget_gate = elementwise_add(forget_gate, bias_ih_forget);
    for (float& val : forget_gate) {
        val = sigmoid(val);
    }

    // Output gate
    std::vector<float> output_gate = matrix_vector_mul(weight_ih_output, input);
    if (output_gate.empty()) return {}; // Return if multiplication failed

    hidden_mul = matrix_vector_mul(weight_hh_output, h);
    if (hidden_mul.empty()) return {}; // Return if multiplication failed

    output_gate = elementwise_add(output_gate, hidden_mul);
    if (output_gate.size() != bias_ih_output.size()) {
        std::cerr << "Error: Dimension mismatch after hidden weight multiplication in output gate (expected " 
                  << bias_ih_output.size() << " but got " << output_gate.size() << ")." << std::endl;
        return {};
    }
    output_gate = elementwise_add(output_gate, bias_ih_output);
    for (float& val : output_gate) {
        val = sigmoid(val);
    }

    // Cell state update (intermediate state)
    std::vector<float> g = matrix_vector_mul(weight_ih_cell, input);
    if (g.empty()) return {}; // Return if multiplication failed

    hidden_mul = matrix_vector_mul(weight_hh_cell, h);
    if (hidden_mul.empty()) return {}; // Return if multiplication failed

    g = elementwise_add(g, hidden_mul);
    if (g.size() != bias_ih_cell.size()) {
        std::cerr << "Error: Dimension mismatch in cell state calculation (expected " << bias_ih_cell.size() << " but got " << g.size() << ")." << std::endl;
        return {};
    }
    g = elementwise_add(g, bias_ih_cell);
    for (float& val : g) {
        val = tanh_func(val);
    }

    // Update cell state c
    if (forget_gate.size() != c.size() || input_gate.size() != g.size()) {
        std::cerr << "Error: Dimension mismatch when updating cell state. Forget gate size: " << forget_gate.size()
                  << ", Cell state size: " << c.size() << ", Input gate size: " << input_gate.size() 
                  << ", Intermediate state size: " << g.size() << "." << std::endl;
        return {};
    }
    c = elementwise_add(elementwise_mul(forget_gate, c), elementwise_mul(input_gate, g));

    // Calculate new hidden state h
    std::vector<float> tanh_c(c.size());
    for (size_t j = 0; j < c.size(); ++j) {
        tanh_c[j] = tanh_func(c[j]);
    }
    if (output_gate.size() != tanh_c.size()) {
        std::cerr << "Error: Dimension mismatch when calculating new hidden state. Output gate size: " 
                  << output_gate.size() << ", tanh(c) size: " << tanh_c.size() << "." << std::endl;
        return {};
    }
    h = elementwise_mul(output_gate, tanh_c);

    // Return updated hidden state
    return h;
}
