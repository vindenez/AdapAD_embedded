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
    
    // Initialize weights and biases with correct dimensions
    weight_ih_input = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(input_size));
    weight_hh_input = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size));
    bias_ih_input = std::vector<float>(4 * hidden_size);
    bias_hh_input = std::vector<float>(4 * hidden_size);

    weight_ih_forget = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(input_size));
    weight_hh_forget = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size));
    bias_ih_forget = std::vector<float>(4 * hidden_size);
    bias_hh_forget = std::vector<float>(4 * hidden_size);

    weight_ih_output = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(input_size));
    weight_hh_output = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size));
    bias_ih_output = std::vector<float>(4 * hidden_size);
    bias_hh_output = std::vector<float>(4 * hidden_size);

    weight_ih_cell = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(input_size));
    weight_hh_cell = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size));
    bias_ih_cell = std::vector<float>(4 * hidden_size);
    bias_hh_cell = std::vector<float>(4 * hidden_size);

    // Initialize with random values (you can use a proper random number generator)
    auto init_weights = [](auto& weights) {
        for (auto& row : weights) {
            for (auto& val : row) {
                val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
    };

    auto init_biases = [](auto& biases) {
        for (auto& val : biases) {
            val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    };

    init_weights(weight_ih_input);
    init_weights(weight_hh_input);
    init_weights(weight_ih_forget);
    init_weights(weight_hh_forget);
    init_weights(weight_ih_output);
    init_weights(weight_hh_output);
    init_weights(weight_ih_cell);
    init_weights(weight_hh_cell);

    init_biases(bias_ih_input);
    init_biases(bias_hh_input);
    init_biases(bias_ih_forget);
    init_biases(bias_hh_forget);
    init_biases(bias_ih_output);
    init_biases(bias_hh_output);
    init_biases(bias_ih_cell);
    init_biases(bias_hh_cell);

    // Print initialization details
    std::cout << "LSTMPredictor initialized with:" << std::endl;
    std::cout << "input_size: " << input_size << std::endl;
    std::cout << "hidden_size: " << hidden_size << std::endl;
    std::cout << "weight_ih_input dimensions: " << weight_ih_input.size() << " x " << weight_ih_input[0].size() << std::endl;
    std::cout << "weight_hh_input dimensions: " << weight_hh_input.size() << " x " << weight_hh_input[0].size() << std::endl;
    std::cout << "weight_ih_forget dimensions: " << weight_ih_forget.size() << " x " << weight_ih_forget[0].size() << std::endl;
    std::cout << "weight_hh_forget dimensions: " << weight_hh_forget.size() << " x " << weight_hh_forget[0].size() << std::endl;
    // ... (print dimensions for other weights and biases) ...
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
    : input_size(weight_ih[0].size()),
      hidden_size(weight_hh[0].size()),
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
        std::cout << "LSTMPredictor forward pass:" << std::endl;
        std::cout << "Input size: " << input.size() << std::endl;
        std::cout << "Previous hidden state size: " << prev_h.size() << std::endl;
        std::cout << "Previous cell state size: " << prev_c.size() << std::endl;
        std::cout << "Weight_ih dimensions: " << weight_ih_input.size() << "x" << weight_ih_input[0].size() << std::endl;
        std::cout << "Weight_hh dimensions: " << weight_hh_input.size() << "x" << weight_hh_input[0].size() << std::endl;

        // Input gate
        std::vector<float> input_gate = matrix_vector_mul(weight_ih_input, input);
        std::cout << "Input gate after weight_ih_input multiplication size: " << input_gate.size() << std::endl;

        std::vector<float> hidden_mul = matrix_vector_mul(weight_hh_input, prev_h);
        std::cout << "Hidden multiplication size: " << hidden_mul.size() << std::endl;

        input_gate = elementwise_add(input_gate, hidden_mul);
        std::cout << "Input gate after adding hidden multiplication size: " << input_gate.size() << std::endl;

        input_gate = elementwise_add(input_gate, bias_ih_input);
        input_gate = elementwise_add(input_gate, bias_hh_input);
        for (float& val : input_gate) {
            val = sigmoid(val);
        }

        // Forget gate
        std::cout << "Calculating forget gate..." << std::endl;
        std::cout << "weight_ih_forget dimensions: " << weight_ih_forget.size() << "x" << (weight_ih_forget.empty() ? 0 : weight_ih_forget[0].size()) << std::endl;
        std::cout << "weight_hh_forget dimensions: " << weight_hh_forget.size() << "x" << (weight_hh_forget.empty() ? 0 : weight_hh_forget[0].size()) << std::endl;
        std::cout << "bias_ih_forget size: " << bias_ih_forget.size() << std::endl;
        std::cout << "bias_hh_forget size: " << bias_hh_forget.size() << std::endl;

        std::vector<float> forget_gate = matrix_vector_mul(weight_ih_forget, input);
        std::cout << "Forget gate after weight_ih_forget multiplication size: " << forget_gate.size() << std::endl;

        std::vector<float> hidden_mul_forget = matrix_vector_mul(weight_hh_forget, prev_h);
        std::cout << "Hidden multiplication size for forget gate: " << hidden_mul_forget.size() << std::endl;

        forget_gate = elementwise_add(forget_gate, hidden_mul_forget);
        forget_gate = elementwise_add(forget_gate, bias_ih_forget);
        forget_gate = elementwise_add(forget_gate, bias_hh_forget);
        for (float& val : forget_gate) {
            val = sigmoid(val);
        }

        // Output gate
        std::cout << "Calculating output gate..." << std::endl;
        std::vector<float> output_gate = matrix_vector_mul(weight_ih_output, input);
        std::cout << "Output gate after weight_ih_output multiplication size: " << output_gate.size() << std::endl;
        hidden_mul = matrix_vector_mul(weight_hh_output, prev_h);
        std::cout << "Hidden multiplication size for output gate: " << hidden_mul.size() << std::endl;
        output_gate = elementwise_add(output_gate, hidden_mul);
        output_gate = elementwise_add(output_gate, bias_ih_output);
        output_gate = elementwise_add(output_gate, bias_hh_output);
        for (float& val : output_gate) {
            val = sigmoid(val);
        }

        // Cell state update
        std::cout << "Calculating cell state update..." << std::endl;
        std::vector<float> g = matrix_vector_mul(weight_ih_cell, input);
        std::cout << "Cell state update after weight_ih_cell multiplication size: " << g.size() << std::endl;
        hidden_mul = matrix_vector_mul(weight_hh_cell, prev_h);
        std::cout << "Hidden multiplication size for cell state update: " << hidden_mul.size() << std::endl;
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

        std::cout << "Output size: " << new_h.size() << std::endl;
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