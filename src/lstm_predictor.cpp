// lstm_predictor.cpp

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

    // Initialize weights and biases with uniform distribution  (PyTorch docs nn.LSTM)
    //u(-sqrt(k), sqrt(k)) where k = 1/hidden_size
    std::random_device rd;
    std::mt19937 gen(rd());

    auto uniform_init = [&](int size) {
        float k = 1.0f / size;
        float range = std::sqrt(k);
        std::uniform_real_distribution<float> dist(-range, range);
        return dist;
    };

    auto init_weights = [&](int fan_out, int fan_in) {
        std::uniform_real_distribution<float> dist = uniform_init(hidden_size);
        std::vector<std::vector<float>> w(fan_out, std::vector<float>(fan_in));
        for (auto& row : w) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
        return w;
    };

    auto init_bias = [&](int size) {
        std::uniform_real_distribution<float> dist = uniform_init(hidden_size);
        std::vector<float> b(size);
        for (auto& val : b) {
            val = dist(gen);
        }
        return b;
    };

    // Initialize LSTM weights and biases for all gates
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

    // Initialize hidden and cell states
    h = std::vector<float>(hidden_size, 0.0f);
    c = std::vector<float>(hidden_size, 0.0f);

    // Debugging statements
    std::cout << "Initialized weight_ih_input with size: " << weight_ih_input.size() << " x " << weight_ih_input[0].size() << std::endl;
    // Repeat for other weights if necessary
}

// Constructor to initialize with weights and biases for each gate
LSTMPredictor::LSTMPredictor(
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
    const std::vector<float>& bias_hh_cell,
    int input_size,
    int hidden_size)
    : input_size(input_size),
      hidden_size(hidden_size),
      num_layers(1),
      lookback_len(1),
      weight_ih_input(weight_ih_input),
      weight_hh_input(weight_hh_input),
      bias_ih_input(bias_ih_input),
      bias_hh_input(bias_hh_input),
      weight_ih_forget(weight_ih_forget),
      weight_hh_forget(weight_hh_forget),
      bias_ih_forget(bias_ih_forget),
      bias_hh_forget(bias_hh_forget),
      weight_ih_output(weight_ih_output),
      weight_hh_output(weight_hh_output),
      bias_ih_output(bias_ih_output),
      bias_hh_output(bias_hh_output),
      weight_ih_cell(weight_ih_cell),
      weight_hh_cell(weight_hh_cell),
      bias_ih_cell(bias_ih_cell),
      bias_hh_cell(bias_hh_cell) {

    // Initialize hidden and cell states
    h = std::vector<float>(hidden_size, 0.0f);
    c = std::vector<float>(hidden_size, 0.0f);
    is_training = false;
}

// Copy constructor implementation
LSTMPredictor::LSTMPredictor(const LSTMPredictor& other)
    : input_size(other.input_size),
      hidden_size(other.hidden_size),
      num_layers(other.num_layers),
      lookback_len(other.lookback_len),
      weight_ih_input(other.weight_ih_input),
      weight_hh_input(other.weight_hh_input),
      bias_ih_input(other.bias_ih_input),
      bias_hh_input(other.bias_hh_input),
      weight_ih_forget(other.weight_ih_forget),
      weight_hh_forget(other.weight_hh_forget),
      bias_ih_forget(other.bias_ih_forget),
      bias_hh_forget(other.bias_hh_forget),
      weight_ih_output(other.weight_ih_output),
      weight_hh_output(other.weight_hh_output),
      bias_ih_output(other.bias_ih_output),
      bias_hh_output(other.bias_hh_output),
      weight_ih_cell(other.weight_ih_cell),
      weight_hh_cell(other.weight_hh_cell),
      bias_ih_cell(other.bias_ih_cell),
      bias_hh_cell(other.bias_hh_cell),
      h(other.h),
      c(other.c) {

    is_training = false;
}

// Getters
int LSTMPredictor::get_input_size() const {
    return input_size;
}

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
            throw std::runtime_error("Input size mismatch in LSTM forward pass.");
        }

        // Input gate
        auto i_t = sigmoid_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_input, input),
                matrix_vector_mul(weight_hh_input, prev_h)),
            bias_ih_input));

        // Forget gate
        auto f_t = sigmoid_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_forget, input),
                matrix_vector_mul(weight_hh_forget, prev_h)),
            bias_ih_forget));

        // Cell gate (candidate cell state)
        auto g_t = tanh_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_cell, input),
                matrix_vector_mul(weight_hh_cell, prev_h)),
            bias_ih_cell));

        // Output gate
        auto o_t = sigmoid_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_output, input),
                matrix_vector_mul(weight_hh_output, prev_h)),
            bias_ih_output));

        // Update cell state
        std::vector<float> new_c(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            new_c[i] = f_t[i] * prev_c[i] + i_t[i] * g_t[i];
        }

        // Calculate new hidden state
        std::vector<float> new_h(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            new_h[i] = o_t[i] * tanh_func(new_c[i]);
        }

        return {new_h, new_h, new_c};
    } catch (const std::exception& e) {
        std::cerr << "Error in LSTM forward pass: " << e.what() << std::endl;
        return {{}, {}, {}};
    }
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>, std::vector<float>>
LSTMPredictor::forward_step(float input, const std::vector<float>& prev_h, const std::vector<float>& prev_c) {
    // Create input vector from input float
    std::vector<float> input_vec = {input};

    // Pad input_vec to match input_size if necessary
    if (input_size > 1) {
        input_vec.resize(input_size, 0.0f);
    }

    // Input gate
    auto i_t = sigmoid_vector(elementwise_add(
        elementwise_add(
            matrix_vector_mul(weight_ih_input, input_vec),
            matrix_vector_mul(weight_hh_input, prev_h)),
        bias_ih_input));

    // Forget gate
    auto f_t = sigmoid_vector(elementwise_add(
        elementwise_add(
            matrix_vector_mul(weight_ih_forget, input_vec),
            matrix_vector_mul(weight_hh_forget, prev_h)),
        bias_ih_forget));

    // Cell gate (candidate cell state)
    auto g_t = tanh_vector(elementwise_add(
        elementwise_add(
            matrix_vector_mul(weight_ih_cell, input_vec),
            matrix_vector_mul(weight_hh_cell, prev_h)),
        bias_ih_cell));

    // Output gate
    auto o_t = sigmoid_vector(elementwise_add(
        elementwise_add(
            matrix_vector_mul(weight_ih_output, input_vec),
            matrix_vector_mul(weight_hh_output, prev_h)),
        bias_ih_output));

    // Update cell state
    std::vector<float> c_t(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        c_t[i] = f_t[i] * prev_c[i] + i_t[i] * g_t[i];
    }

    // Calculate new hidden state
    std::vector<float> h_t(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        h_t[i] = o_t[i] * tanh_func(c_t[i]);
    }

    return {i_t, f_t, o_t, g_t, c_t, h_t};
}

void LSTMPredictor::train() {
    is_training = true;
}

void LSTMPredictor::eval() {
    is_training = false;
}

void LSTMPredictor::zero_grad() {
    dw_ih_input = std::vector<std::vector<float>>(weight_ih_input.size(), std::vector<float>(weight_ih_input[0].size(), 0.0f));
    dw_hh_input = std::vector<std::vector<float>>(weight_hh_input.size(), std::vector<float>(weight_hh_input[0].size(), 0.0f));
    db_ih_input = std::vector<float>(bias_ih_input.size(), 0.0f);
    db_hh_input = std::vector<float>(bias_hh_input.size(), 0.0f);
}

std::vector<float> LSTMPredictor::forward(const std::vector<float>& input) {
    std::vector<float> output;
    std::vector<float> h_t = h;
    std::vector<float> c_t = c;

    for (float x : input) {
        auto [i_t, f_t, o_t, g_t, new_c, new_h] = forward_step(x, h_t, c_t);
        h_t = new_h;
        c_t = new_c;
        output.push_back(new_h[0]);
    }

    if (is_training) {
        h = h_t;
        c = c_t;
    }

    return output;
}

const std::vector<float>& LSTMPredictor::get_h() const {
    return h;
}

const std::vector<float>& LSTMPredictor::get_c() const {
    return c;
}
