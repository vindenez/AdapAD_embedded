// lstm_predictor.cpp

#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Constructor to initialize with input size, hidden size, and other hyperparameters
LSTMPredictor::LSTMPredictor(int input_size, int hidden_size, int output_size, int num_layers, int lookback_len)
    : input_size(1),  // Force single feature input
      hidden_size(hidden_size),
      output_size(1), // Force single prediction output
      num_layers(num_layers),
      lookback_len(lookback_len) {

    // Initialize weights with proper dimensions and smaller initial values
    auto init_gate_weights = [&](int fan_out, int fan_in) {
        std::random_device rd;
        std::mt19937 gen(rd());
        // Use Xavier/Glorot initialization with smaller scale
        float scale = 0.1f * std::sqrt(2.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-scale, scale);
        
        std::vector<std::vector<float>> w(fan_out, std::vector<float>(fan_in));
        for (auto& row : w) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
        return w;
    };

    // Initialize gate weights with correct dimensions
    weight_ih_input = init_gate_weights(hidden_size, 1);
    weight_hh_input = init_gate_weights(hidden_size, hidden_size);
    
    weight_ih_forget = init_gate_weights(hidden_size, 1);
    weight_hh_forget = init_gate_weights(hidden_size, hidden_size);
    
    weight_ih_cell = init_gate_weights(hidden_size, 1);
    weight_hh_cell = init_gate_weights(hidden_size, hidden_size);
    
    weight_ih_output = init_gate_weights(hidden_size, 1);
    weight_hh_output = init_gate_weights(hidden_size, hidden_size);

    // Initialize fully connected layer with smaller weights
    fc_weights = init_gate_weights(1, hidden_size);
    fc_bias = std::vector<float>(1, 0.0f);

    // Initialize biases with small positive values for forget gate
    bias_ih_input = std::vector<float>(hidden_size, 0.0f);
    bias_hh_input = std::vector<float>(hidden_size, 0.0f);
    
    bias_ih_forget = std::vector<float>(hidden_size, 1.0f);  // Initialize forget gate bias to 1
    bias_hh_forget = std::vector<float>(hidden_size, 1.0f);  // This helps with gradient flow
    
    bias_ih_cell = std::vector<float>(hidden_size, 0.0f);
    bias_hh_cell = std::vector<float>(hidden_size, 0.0f);
    
    bias_ih_output = std::vector<float>(hidden_size, 0.0f);
    bias_hh_output = std::vector<float>(hidden_size, 0.0f);

    // Initialize states
    h = std::vector<float>(hidden_size, 0.0f);
    c = std::vector<float>(hidden_size, 0.0f);
    h_init = h;
    c_init = c;
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

        std::vector<float> output = matrix_vector_mul(fc_weights, new_h);
        output = elementwise_add(output, fc_bias);

        return {output, new_h, new_c};
    } catch (const std::exception& e) {
        std::cerr << "Error in LSTM forward pass: " << e.what() << std::endl;
        return {{}, {}, {}};
    }
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
LSTMPredictor::forward_step(const std::vector<float>& x_t, const std::vector<float>& prev_h, const std::vector<float>& prev_c) {
    try {
        // Ensure input is single feature
        if (x_t.size() != 1) {
            throw std::runtime_error("Input size mismatch. Expected 1, got " + std::to_string(x_t.size()));
        }

        // Input gate
        auto i_t = sigmoid_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_input, x_t),
                matrix_vector_mul(weight_hh_input, prev_h)),
            bias_ih_input));

        // Forget gate
        auto f_t = sigmoid_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_forget, x_t),
                matrix_vector_mul(weight_hh_forget, prev_h)),
            bias_ih_forget));

        // Cell gate
        auto g_t = tanh_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_cell, x_t),
                matrix_vector_mul(weight_hh_cell, prev_h)),
            bias_ih_cell));

        // Output gate
        auto o_t = sigmoid_vector(elementwise_add(
            elementwise_add(
                matrix_vector_mul(weight_ih_output, x_t),
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

        // Apply fully connected layer to new hidden state
        std::vector<float> output = matrix_vector_mul(fc_weights, h_t);
        output = elementwise_add(output, fc_bias);

        return {i_t, f_t, o_t, g_t, c_t, h_t, output};
    } catch (const std::exception& e) {
        std::cerr << "Error in LSTM forward_step: " << e.what() << std::endl;
        throw;
    }
}


void LSTMPredictor::train() {
    is_training = true;
}

void LSTMPredictor::eval() {
    is_training = false;
}

void LSTMPredictor::zero_grad() {
    // Zero gradients for Input Gate
    dw_ih_input = std::vector<std::vector<float>>(hidden_size, std::vector<float>(input_size, 0.0f));
    dw_hh_input = std::vector<std::vector<float>>(hidden_size, std::vector<float>(hidden_size, 0.0f));
    db_ih_input = std::vector<float>(hidden_size, 0.0f);
    db_hh_input = std::vector<float>(hidden_size, 0.0f);

    // Zero gradients for Forget Gate
    dw_ih_forget = std::vector<std::vector<float>>(hidden_size, std::vector<float>(input_size, 0.0f));
    dw_hh_forget = std::vector<std::vector<float>>(hidden_size, std::vector<float>(hidden_size, 0.0f));
    db_ih_forget = std::vector<float>(hidden_size, 0.0f);
    db_hh_forget = std::vector<float>(hidden_size, 0.0f);

    // Zero gradients for Output Gate
    dw_ih_output = std::vector<std::vector<float>>(hidden_size, std::vector<float>(input_size, 0.0f));
    dw_hh_output = std::vector<std::vector<float>>(hidden_size, std::vector<float>(hidden_size, 0.0f));
    db_ih_output = std::vector<float>(hidden_size, 0.0f);
    db_hh_output = std::vector<float>(hidden_size, 0.0f);

    // Zero gradients for Cell Gate
    dw_ih_cell = std::vector<std::vector<float>>(hidden_size, std::vector<float>(input_size, 0.0f));
    dw_hh_cell = std::vector<std::vector<float>>(hidden_size, std::vector<float>(hidden_size, 0.0f));
    db_ih_cell = std::vector<float>(hidden_size, 0.0f);
    db_hh_cell = std::vector<float>(hidden_size, 0.0f);

    // Zero gradients for Fully Connected Layer
    dw_fc_weights = std::vector<std::vector<float>>(fc_weights.size(), std::vector<float>(fc_weights[0].size(), 0.0f));
    db_fc_bias = std::vector<float>(fc_bias.size(), 0.0f);
}


std::vector<float> LSTMPredictor::forward(const std::vector<std::vector<std::vector<float>>>& input_sequence) {
    // Clear stored activations
    x_t_list.clear();
    i_t_list.clear();
    f_t_list.clear();
    o_t_list.clear();
    g_t_list.clear();
    c_t_list.clear();
    h_t_list.clear();
    outputs_list.clear();

    // Initialize hidden and cell states
    if (h.empty()) {
        h = std::vector<float>(hidden_size, 0.0f);
    }
    if (c.empty()) {
        c = std::vector<float>(hidden_size, 0.0f);
    }
    
    std::vector<float> h_t = h;
    std::vector<float> c_t = c;

    // Store initial states
    h_init = h_t;
    c_init = c_t;

    std::vector<float> outputs;

    // Process each timestep in the sequence
    const auto& batch = input_sequence[0];  // First batch
    for (const auto& x_t : batch) {
        x_t_list.push_back(x_t);

        // Process one time step
        auto [i_t, f_t, o_t, g_t, new_c, new_h, output] = forward_step(x_t, h_t, c_t);

        // Update states
        h_t = new_h;
        c_t = new_c;

        // Store activations
        i_t_list.push_back(i_t);
        f_t_list.push_back(f_t);
        o_t_list.push_back(o_t);
        g_t_list.push_back(g_t);
        c_t_list.push_back(new_c);
        h_t_list.push_back(new_h);
        outputs_list.push_back(output);

        outputs.push_back(output[0]);
    }

    if (is_training) {
        h = h_t;
        c = c_t;
    }

    return outputs;
}

void LSTMPredictor::backward(const std::vector<float>& targets, const std::string& loss_function) {
    // Initialize gradients to zero
    zero_grad();

    // Compute loss gradient at the output layer
    std::vector<std::vector<float>> d_output_list(outputs_list.size());

    for (size_t t = 0; t < outputs_list.size(); ++t) {
        float output = outputs_list[t][0];  // Assuming output_size == 1
        float target = targets[t];
        float error = output - target;

        // Compute gradient of loss w.r.t. output
        float d_loss_d_output;
        if (loss_function == "MSE") {
            d_loss_d_output = 2.0f * error;
        } else {
            throw std::runtime_error("Unsupported loss function");
        }

        d_output_list[t] = {d_loss_d_output};  // Assuming output_size == 1
    }

    // Initialize gradients for hidden and cell states
    std::vector<float> dh_next(hidden_size, 0.0f);
    std::vector<float> dc_next(hidden_size, 0.0f);

    // Backpropagate through time
    for (int t = outputs_list.size() - 1; t >= 0; --t) {
        // Get stored activations
        auto& x_t = x_t_list[t];
        auto& i_t = i_t_list[t];
        auto& f_t = f_t_list[t];
        auto& o_t = o_t_list[t];
        auto& g_t = g_t_list[t];
        auto& c_t = c_t_list[t];
        auto& h_t = h_t_list[t];

        // Get previous hidden and cell states
        std::vector<float> h_prev, c_prev;
        if (t > 0) {
            h_prev = h_t_list[t - 1];
            c_prev = c_t_list[t - 1];
        } else {
            h_prev = h_init;
            c_prev = c_init;
        }

        // Compute gradients at fully connected layer
        auto& d_output = d_output_list[t];

        // Gradients w.r.t. fully connected layer parameters
        for (size_t i = 0; i < fc_weights.size(); ++i) {
            for (size_t j = 0; j < fc_weights[0].size(); ++j) {
                dw_fc_weights[i][j] += d_output[i] * h_t[j];
            }
            db_fc_bias[i] += d_output[i];
        }

        // Backpropagate to hidden state
        std::vector<float> dh = matrix_vector_mul(transpose_matrix(fc_weights), d_output);

        // Add gradients from next time step
        for (size_t i = 0; i < dh.size(); ++i) {
            dh[i] += dh_next[i];
        }

        // Compute dc_t
        std::vector<float> dc = dc_next;

        // tanh_c_t = tanh(c_t)
        std::vector<float> tanh_c_t(c_t.size());
        for (size_t i = 0; i < c_t.size(); ++i) {
            tanh_c_t[i] = tanh_func(c_t[i]);
        }

        // Compute do_t = dh * tanh(c_t) * o_t * (1 - o_t)
        std::vector<float> do_t(o_t.size());
        for (size_t i = 0; i < o_t.size(); ++i) {
            do_t[i] = dh[i] * tanh_c_t[i] * o_t[i] * (1 - o_t[i]);
        }

        // Compute dc += dh * o_t * (1 - tanh(c_t)^2)
        for (size_t i = 0; i < dc.size(); ++i) {
            dc[i] += dh[i] * o_t[i] * (1 - tanh_c_t[i] * tanh_c_t[i]);
        }

        // Compute di_t = dc * g_t * i_t * (1 - i_t)
        std::vector<float> di_t(i_t.size());
        for (size_t i = 0; i < i_t.size(); ++i) {
            di_t[i] = dc[i] * g_t[i] * i_t[i] * (1 - i_t[i]);
        }

        // Compute df_t = dc * c_prev * f_t * (1 - f_t)
        std::vector<float> df_t(f_t.size());
        for (size_t i = 0; i < f_t.size(); ++i) {
            df_t[i] = dc[i] * c_prev[i] * f_t[i] * (1 - f_t[i]);
        }

        // Compute dg_t = dc * i_t * (1 - g_t^2)
        std::vector<float> dg_t(g_t.size());
        for (size_t i = 0; i < g_t.size(); ++i) {
            dg_t[i] = dc[i] * i_t[i] * (1 - g_t[i] * g_t[i]);
        }

        // Compute gradients w.r.t. input weights
        for (size_t i = 0; i < hidden_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                dw_ih_input[i][j] += di_t[i] * x_t[j];
                dw_ih_forget[i][j] += df_t[i] * x_t[j];
                dw_ih_output[i][j] += do_t[i] * x_t[j];
                dw_ih_cell[i][j] += dg_t[i] * x_t[j];
            }
        }

        // Compute gradients w.r.t. hidden weights
        for (size_t i = 0; i < hidden_size; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                dw_hh_input[i][j] += di_t[i] * h_prev[j];
                dw_hh_forget[i][j] += df_t[i] * h_prev[j];
                dw_hh_output[i][j] += do_t[i] * h_prev[j];
                dw_hh_cell[i][j] += dg_t[i] * h_prev[j];
            }
        }

        // Compute gradients w.r.t. biases (input and hidden biases)
        for (size_t i = 0; i < hidden_size; ++i) {
            db_ih_input[i] += di_t[i];
            db_ih_forget[i] += df_t[i];
            db_ih_output[i] += do_t[i];
            db_ih_cell[i] += dg_t[i];

            db_hh_input[i] += di_t[i];
            db_hh_forget[i] += df_t[i];
            db_hh_output[i] += do_t[i];
            db_hh_cell[i] += dg_t[i];
        }

        // Compute dh_prev
        std::vector<float> dh_prev(hidden_size, 0.0f);
        dh_prev = elementwise_add(
            elementwise_add(
                elementwise_add(
                    matrix_vector_mul(transpose_matrix(weight_hh_input), di_t),
                    matrix_vector_mul(transpose_matrix(weight_hh_forget), df_t)),
                matrix_vector_mul(transpose_matrix(weight_hh_output), do_t)),
            matrix_vector_mul(transpose_matrix(weight_hh_cell), dg_t)
        );

        // Compute dc_prev
        std::vector<float> dc_prev(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            dc_prev[i] = dc[i] * f_t[i];
        }

        // Update dh_next and dc_next for next time step
        dh_next = dh_prev;
        dc_next = dc_prev;
    }
}


void LSTMPredictor::init_adam_optimizer(float learning_rate) {
    this->learning_rate = learning_rate;
    beta1 = 0.9f;
    beta2 = 0.999f;
    epsilon = 1e-8f;
    t = 0;

    // Initialize first and second moment estimates to zero
    auto init_moment = [](const std::vector<std::vector<float>>& weights) {
        return std::vector<std::vector<float>>(weights.size(), std::vector<float>(weights[0].size(), 0.0f));
    };

    auto init_bias_moment = [](const std::vector<float>& biases) {
        return std::vector<float>(biases.size(), 0.0f);
    };

    // Input Gate
    m_w_ih_input = init_moment(weight_ih_input);
    m_w_hh_input = init_moment(weight_hh_input);
    m_b_ih_input = init_bias_moment(bias_ih_input);
    m_b_hh_input = init_bias_moment(bias_hh_input);

    v_w_ih_input = init_moment(weight_ih_input);
    v_w_hh_input = init_moment(weight_hh_input);
    v_b_ih_input = init_bias_moment(bias_ih_input);
    v_b_hh_input = init_bias_moment(bias_hh_input);

    // Forget Gate
    m_w_ih_forget = init_moment(weight_ih_forget);
    m_w_hh_forget = init_moment(weight_hh_forget);
    m_b_ih_forget = init_bias_moment(bias_ih_forget);
    m_b_hh_forget = init_bias_moment(bias_hh_forget);

    v_w_ih_forget = init_moment(weight_ih_forget);
    v_w_hh_forget = init_moment(weight_hh_forget);
    v_b_ih_forget = init_bias_moment(bias_ih_forget);
    v_b_hh_forget = init_bias_moment(bias_hh_forget);

    // Output Gate
    m_w_ih_output = init_moment(weight_ih_output);
    m_w_hh_output = init_moment(weight_hh_output);
    m_b_ih_output = init_bias_moment(bias_ih_output);
    m_b_hh_output = init_bias_moment(bias_hh_output);

    v_w_ih_output = init_moment(weight_ih_output);
    v_w_hh_output = init_moment(weight_hh_output);
    v_b_ih_output = init_bias_moment(bias_ih_output);
    v_b_hh_output = init_bias_moment(bias_hh_output);

    // Cell Gate
    m_w_ih_cell = init_moment(weight_ih_cell);
    m_w_hh_cell = init_moment(weight_hh_cell);
    m_b_ih_cell = init_bias_moment(bias_ih_cell);
    m_b_hh_cell = init_bias_moment(bias_hh_cell);

    v_w_ih_cell = init_moment(weight_ih_cell);
    v_w_hh_cell = init_moment(weight_hh_cell);
    v_b_ih_cell = init_bias_moment(bias_ih_cell);
    v_b_hh_cell = init_bias_moment(bias_hh_cell);

    // Fully Connected Layer
    m_fc_weights = init_moment(fc_weights);
    m_fc_bias = init_bias_moment(fc_bias);

    v_fc_weights = init_moment(fc_weights);
    v_fc_bias = init_bias_moment(fc_bias);
}

void LSTMPredictor::update_parameters_adam(float learning_rate) {
    t++;
    float alpha_t = learning_rate * std::sqrt(1 - std::pow(beta2, t)) / (1 - std::pow(beta1, t));

    // Function to update parameters
    auto update_parameters = [&](std::vector<std::vector<float>>& weights,
                                 std::vector<std::vector<float>>& m_weights,
                                 std::vector<std::vector<float>>& v_weights,
                                 const std::vector<std::vector<float>>& dw,
                                 std::vector<float>& biases,
                                 std::vector<float>& m_biases,
                                 std::vector<float>& v_biases,
                                 const std::vector<float>& db) {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                // Update biased first moment estimate
                m_weights[i][j] = beta1 * m_weights[i][j] + (1 - beta1) * dw[i][j];

                // Update biased second raw moment estimate
                v_weights[i][j] = beta2 * v_weights[i][j] + (1 - beta2) * dw[i][j] * dw[i][j];

                // Compute bias-corrected estimates
                float m_hat = m_weights[i][j] / (1 - std::pow(beta1, t));
                float v_hat = v_weights[i][j] / (1 - std::pow(beta2, t));

                // Update parameters
                weights[i][j] -= alpha_t * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }

        for (size_t i = 0; i < biases.size(); ++i) {
            // Update biased first moment estimate
            m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * db[i];

            // Update biased second raw moment estimate
            v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * db[i] * db[i];

            // Compute bias-corrected estimates
            float m_hat = m_biases[i] / (1 - std::pow(beta1, t));
            float v_hat = v_biases[i] / (1 - std::pow(beta2, t));

            // Update parameters
            biases[i] -= alpha_t * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    };

    // Update parameters for Input Gate
    update_parameters(weight_ih_input, m_w_ih_input, v_w_ih_input, dw_ih_input,
                      bias_ih_input, m_b_ih_input, v_b_ih_input, db_ih_input);
    update_parameters(weight_hh_input, m_w_hh_input, v_w_hh_input, dw_hh_input,
                      bias_hh_input, m_b_hh_input, v_b_hh_input, db_hh_input);

    // Update parameters for Forget Gate
    update_parameters(weight_ih_forget, m_w_ih_forget, v_w_ih_forget, dw_ih_forget,
                      bias_ih_forget, m_b_ih_forget, v_b_ih_forget, db_ih_forget);
    update_parameters(weight_hh_forget, m_w_hh_forget, v_w_hh_forget, dw_hh_forget,
                      bias_hh_forget, m_b_hh_forget, v_b_hh_forget, db_hh_forget);

    // Update parameters for Output Gate
    update_parameters(weight_ih_output, m_w_ih_output, v_w_ih_output, dw_ih_output,
                      bias_ih_output, m_b_ih_output, v_b_ih_output, db_ih_output);
    update_parameters(weight_hh_output, m_w_hh_output, v_w_hh_output, dw_hh_output,
                      bias_hh_output, m_b_hh_output, v_b_hh_output, db_hh_output);

    // Update parameters for Cell Gate
    update_parameters(weight_ih_cell, m_w_ih_cell, v_w_ih_cell, dw_ih_cell,
                      bias_ih_cell, m_b_ih_cell, v_b_ih_cell, db_ih_cell);
    update_parameters(weight_hh_cell, m_w_hh_cell, v_w_hh_cell, dw_hh_cell,
                      bias_hh_cell, m_b_hh_cell, v_b_hh_cell, db_hh_cell);

    // Update parameters for Fully Connected Layer
    update_parameters(fc_weights, m_fc_weights, v_fc_weights, dw_fc_weights,
                      fc_bias, m_fc_bias, v_fc_bias, db_fc_bias);
}

const std::vector<float>& LSTMPredictor::get_h() const {
    return h;
}

const std::vector<float>& LSTMPredictor::get_c() const {
    return c;
}
