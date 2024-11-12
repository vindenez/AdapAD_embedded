#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <config.hpp>
#include <algorithm>

// Constructor to initialize with input size, hidden size, and other hyperparameters
LSTMPredictor::LSTMPredictor(int input_size, int hidden_size, int output_size, int num_layers, int lookback_len)
    : input_size(lookback_len),
      hidden_size(hidden_size),
      output_size(output_size),
      num_layers(num_layers),
      lookback_len(lookback_len) {

    // Initialize weight parameters
    float k = 1.0f / std::sqrt(hidden_size);
    std::uniform_real_distribution<float> dist(-k, k);
    auto weight_init = [this, &dist](float) { return dist(gen); };

    // Initialize weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        // Initialize input-hidden weights
        if (layer == 0) {
            w_ih.push_back(init_weights(4 * hidden_size, input_size, weight_init));
        } else {
            w_ih.push_back(init_weights(4 * hidden_size, hidden_size, weight_init));
        }
        
        // Initialize hidden-hidden weights
        w_hh.push_back(init_weights(4 * hidden_size, hidden_size, weight_init));
        
        // Initialize biases
        b_ih.push_back(std::vector<float>(4 * hidden_size, 0.0f));
        b_hh.push_back(std::vector<float>(4 * hidden_size, 0.0f));
        
        // Initialize forget gate biases to 1.0 (PyTorch default)
        for (int i = hidden_size; i < 2 * hidden_size; i++) {
            b_ih.back()[i] = 1.0f;
            b_hh.back()[i] = 1.0f;
        }
    }
    
    // Initialize fully connected layer
    fc_weights = init_weights(output_size, hidden_size, weight_init);
    fc_bias = std::vector<float>(output_size, 0.0f);
    
    // Initialize states
    h_states = std::vector<std::vector<float>>(num_layers, std::vector<float>(hidden_size, 0.0f));
    c_states = std::vector<std::vector<float>>(num_layers, std::vector<float>(hidden_size, 0.0f));
    
    // Initialize gradients
    dw_ih = std::vector<std::vector<std::vector<float>>>(num_layers);
    dw_hh = std::vector<std::vector<std::vector<float>>>(num_layers);
    db_ih = std::vector<std::vector<float>>(num_layers);
    db_hh = std::vector<std::vector<float>>(num_layers);
    
    zero_grad();
}

std::vector<std::vector<float>> LSTMPredictor::init_weights(int rows, int cols, 
                                                           const std::function<float(float)>& init_func) {
    std::vector<std::vector<float>> weights(rows, std::vector<float>(cols));
    for (auto& row : weights) {
        for (auto& val : row) {
            val = init_func(val);
        }
    }
    return weights;
}

void LSTMPredictor::train() {
    is_training = true;
}

void LSTMPredictor::eval() {
    is_training = false;
}

void LSTMPredictor::zero_grad() {
    // Zero out layer gradients
    for (int layer = 0; layer < num_layers; ++layer) {
        if (layer == 0) {
            dw_ih[layer] = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(input_size, 0.0f));
        } else {
            dw_ih[layer] = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        }
        dw_hh[layer] = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        db_ih[layer] = std::vector<float>(4 * hidden_size, 0.0f);
        db_hh[layer] = std::vector<float>(4 * hidden_size, 0.0f);
    }

    // Zero out fully connected layer gradients
    dw_fc = std::vector<std::vector<float>>(output_size, std::vector<float>(hidden_size, 0.0f));
    db_fc = std::vector<float>(output_size, 0.0f);
}


void LSTMPredictor::backward(const std::vector<float>& targets, const std::string& loss_function) {
    if (!is_training) {
        throw std::runtime_error("Backward called while not in training mode");
    }
    
    // Initialize gradients
    std::vector<float> grad_output(output_size);
    std::string loss_lower = loss_function;
    std::transform(loss_lower.begin(), loss_lower.end(), loss_lower.begin(), ::tolower);
    
    if (loss_lower == "mse") {
        const auto& final_hidden = h_states.back();
        if (final_hidden.empty()) {
            throw std::runtime_error("No hidden states available for backward pass");
        }
        
        // Compute gradients with respect to the differences
        const auto& last_input = layer_inputs.back().back();
        for (size_t i = 0; i < output_size; i++) {
            float pred_diff = final_hidden[i] - last_input[i];
            float target_diff = targets[i] - last_input[i];
            grad_output[i] = 2.0f * (pred_diff - target_diff) / output_size;
        }
    } else {
        throw std::runtime_error("Unsupported loss function: " + loss_function);
    }
    
    // Backward through fully connected layer
    auto dh = matrix_vector_mul_transpose(fc_weights, grad_output);
    const auto& final_hidden = h_states.back();
    dw_fc = outer_product(grad_output, final_hidden);
    db_fc = grad_output;
    
    // Backward through LSTM layers
    for (int layer = num_layers - 1; layer >= 0; --layer) {
        lstm_layer_backward(layer, dh);
    }
}

void LSTMPredictor::lstm_layer_backward(int layer, const std::vector<float>& grad_output) {
    int seq_len = layer_inputs[layer].size();
    std::vector<float> dh_next = grad_output;
    std::vector<float> dc_next(hidden_size, 0.0f);
    
    // Initialize layer gradients
    dw_ih[layer] = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(layer == 0 ? input_size : hidden_size, 0.0f));
    dw_hh[layer] = std::vector<std::vector<float>>(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
    db_ih[layer] = std::vector<float>(4 * hidden_size, 0.0f);
    db_hh[layer] = std::vector<float>(4 * hidden_size, 0.0f);
    
    // Iterate backwards through time
    for (int t = seq_len - 1; t >= 0; --t) {
        const auto& gates = layer_gates[layer][t];
        const auto& prev_c = layer_c_states[layer][t];
        const auto& prev_h = layer_h_states[layer][t];
        const auto& x = layer_inputs[layer][t];
        
        // Split gates
        int chunk_size = hidden_size;
        auto i_gate = std::vector<float>(gates.begin(), gates.begin() + chunk_size);
        auto f_gate = std::vector<float>(gates.begin() + chunk_size, gates.begin() + 2 * chunk_size);
        auto g_gate = std::vector<float>(gates.begin() + 2 * chunk_size, gates.begin() + 3 * chunk_size);
        auto o_gate = std::vector<float>(gates.begin() + 3 * chunk_size, gates.end());
        
        // Current cell state
        auto c = c_states[layer];
        
        // Gradients of hidden and cell states
        auto dh = dh_next;
        auto dc = dc_next;
        
        // Output gate
        auto do_gate = elementwise_mul(dh, tanh_vector(c));
        do_gate = elementwise_mul(do_gate, sigmoid_derivative(o_gate));
        
        // Cell state
        dc = elementwise_add(dc, elementwise_mul(dh, elementwise_mul(o_gate, tanh_derivative(c))));
        
        // Input gate
        auto di_gate = elementwise_mul(dc, g_gate);
        di_gate = elementwise_mul(di_gate, sigmoid_derivative(i_gate));
        
        // Forget gate
        auto df_gate = elementwise_mul(dc, prev_c);
        df_gate = elementwise_mul(df_gate, sigmoid_derivative(f_gate));
        
        // Cell gate
        auto dg_gate = elementwise_mul(dc, i_gate);
        dg_gate = elementwise_mul(dg_gate, tanh_derivative(g_gate));
        
        // Concatenate gate gradients
        std::vector<float> dgates;
        dgates.insert(dgates.end(), di_gate.begin(), di_gate.end());
        dgates.insert(dgates.end(), df_gate.begin(), df_gate.end());
        dgates.insert(dgates.end(), dg_gate.begin(), dg_gate.end());
        dgates.insert(dgates.end(), do_gate.begin(), do_gate.end());
        
        // Update weight gradients
        auto dw_ih_t = outer_product(dgates, x);
        auto dw_hh_t = outer_product(dgates, prev_h);
        
        // Accumulate gradients
        dw_ih[layer] = matrix_add(dw_ih[layer], dw_ih_t);
        dw_hh[layer] = matrix_add(dw_hh[layer], dw_hh_t);
        db_ih[layer] = elementwise_add(db_ih[layer], dgates);
        db_hh[layer] = elementwise_add(db_hh[layer], dgates);
        
        // Compute gradients for previous timestep
        dh_next = matrix_vector_mul_transpose(w_hh[layer], dgates);
        dc_next = elementwise_mul(dc, f_gate);
    }
}

void LSTMPredictor::update_parameters_adam(float learning_rate) {
    lr = learning_rate;
    
    // Lambda for updating weights (2D)
    auto update_weights = [&](std::vector<std::vector<float>>& params,
                            const std::vector<std::vector<float>>& grad,
                            AdamState& state) {
        state.step += 1;
        float bias_correction1 = 1.0f - std::pow(beta1, state.step);
        float bias_correction2 = 1.0f - std::pow(beta2, state.step);
        
        for (size_t i = 0; i < params.size(); ++i) {
            for (size_t j = 0; j < params[i].size(); ++j) {
                float grad_val = grad[i][j];
                if (weight_decay != 0.0f) {
                    grad_val += weight_decay * params[i][j];
                }
                
                // Update first and second moments
                state.exp_avg[i][j] = beta1 * state.exp_avg[i][j] + (1.0f - beta1) * grad_val;
                state.exp_avg_sq[i][j] = beta2 * state.exp_avg_sq[i][j] + (1.0f - beta2) * grad_val * grad_val;
                
                // Compute bias-corrected moments
                float exp_avg_corrected = state.exp_avg[i][j] / bias_correction1;
                float exp_avg_sq_corrected = state.exp_avg_sq[i][j] / bias_correction2;
                
                // Update parameters
                params[i][j] -= lr * exp_avg_corrected / (std::sqrt(exp_avg_sq_corrected) + epsilon);
            }
        }
    };

    // Lambda for updating biases (1D)
    auto update_biases = [&](std::vector<float>& params,
                           const std::vector<float>& grad,
                           AdamState& state) {
        state.step += 1;
        float bias_correction1 = 1.0f - std::pow(beta1, state.step);
        float bias_correction2 = 1.0f - std::pow(beta2, state.step);
        
        for (size_t i = 0; i < params.size(); ++i) {
            float grad_val = grad[i];
            if (weight_decay != 0.0f) {
                grad_val += weight_decay * params[i];
            }
            
            // Update first and second moments
            state.exp_avg[0][i] = beta1 * state.exp_avg[0][i] + (1.0f - beta1) * grad_val;
            state.exp_avg_sq[0][i] = beta2 * state.exp_avg_sq[0][i] + (1.0f - beta2) * grad_val * grad_val;
            
            // Compute bias-corrected moments
            float exp_avg_corrected = state.exp_avg[0][i] / bias_correction1;
            float exp_avg_sq_corrected = state.exp_avg_sq[0][i] / bias_correction2;
            
            // Update parameters
            params[i] -= lr * exp_avg_corrected / (std::sqrt(exp_avg_sq_corrected) + epsilon);
        }
    };

    // Update layer parameters
    for (int layer = 0; layer < num_layers; ++layer) {
        update_weights(w_ih[layer], dw_ih[layer], adam_states["w_ih_" + std::to_string(layer)]);
        update_weights(w_hh[layer], dw_hh[layer], adam_states["w_hh_" + std::to_string(layer)]);
        update_biases(b_ih[layer], db_ih[layer], adam_states["b_ih_" + std::to_string(layer)]);
        update_biases(b_hh[layer], db_hh[layer], adam_states["b_hh_" + std::to_string(layer)]);
    }

    // Update fully connected layer
    update_weights(fc_weights, dw_fc, adam_states["fc_weights"]);
    update_biases(fc_bias, db_fc, adam_states["fc_bias"]);
}

float LSTMPredictor::compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::runtime_error("Output and target size mismatch in MSE loss calculation");
    }
    
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float error = output[i] - target[i];
        loss += error * error;
    }
    return loss / output.size();
}

std::vector<float> LSTMPredictor::apply_dropout(const std::vector<float>& input) {
    if (!is_training) return input;
    
    std::vector<float> output = input;
    std::bernoulli_distribution d(1.0f - dropout_rate);
    
    for (auto& val : output) {
        if (!d(gen)) {
            val = 0;
        } else {
            val /= (1.0f - dropout_rate);  // Scale during training
        }
    }
    return output;
}

std::vector<float> LSTMPredictor::lstm_layer_forward(const std::vector<float>& x, int layer) {
    // Get previous states for this layer
    const auto& prev_h = h_states[layer];
    const auto& prev_c = c_states[layer];
    
    // Compute gates in one matrix multiplication
    auto gates_ih = matrix_vector_mul(w_ih[layer], x);
    auto gates_hh = matrix_vector_mul(w_hh[layer], prev_h);
    
    // Add biases
    auto gates = elementwise_add(elementwise_add(gates_ih, gates_hh), 
                               elementwise_add(b_ih[layer], b_hh[layer]));
    
    // Split gates into chunks
    int chunk_size = hidden_size;
    auto i_gate = sigmoid_vector(std::vector<float>(gates.begin(), 
                                                  gates.begin() + chunk_size));
    auto f_gate = sigmoid_vector(std::vector<float>(gates.begin() + chunk_size, 
                                                  gates.begin() + 2 * chunk_size));
    auto g_gate = tanh_vector(std::vector<float>(gates.begin() + 2 * chunk_size, 
                                               gates.begin() + 3 * chunk_size));
    auto o_gate = sigmoid_vector(std::vector<float>(gates.begin() + 3 * chunk_size, 
                                                  gates.end()));
    
    // Update cell state
    std::vector<float> new_c(hidden_size);
    for (size_t i = 0; i < hidden_size; i++) {
        new_c[i] = f_gate[i] * prev_c[i] + i_gate[i] * g_gate[i];
    }
    
    // Update hidden state
    std::vector<float> new_h(hidden_size);
    for (size_t i = 0; i < hidden_size; i++) {
        new_h[i] = o_gate[i] * tanh_func(new_c[i]);
    }
    
    // Store states and activations for backward pass
    if (is_training) {
        layer_inputs[layer].push_back(x);
        layer_h_states[layer].push_back(prev_h);
        layer_c_states[layer].push_back(prev_c);
        layer_gates[layer].push_back(gates);
    }
    
    // Update states
    h_states[layer] = new_h;
    c_states[layer] = new_c;
    
    return new_h;
}

std::vector<float> LSTMPredictor::forward(const std::vector<std::vector<std::vector<float>>>& input) {
    if (is_training) {
        // Clear stored activations
        layer_inputs = std::vector<std::vector<std::vector<float>>>(num_layers);
        layer_h_states = std::vector<std::vector<std::vector<float>>>(num_layers);
        layer_c_states = std::vector<std::vector<std::vector<float>>>(num_layers);
        layer_gates = std::vector<std::vector<std::vector<float>>>(num_layers);
    }
    
    // Reset states at the start of sequence
    h_states = std::vector<std::vector<float>>(num_layers, std::vector<float>(hidden_size, 0.0f));
    c_states = std::vector<std::vector<float>>(num_layers, std::vector<float>(hidden_size, 0.0f));
    
    std::vector<float> output;
    
    // Process each timestep
    for (const auto& timestep : input) {
        std::vector<float> layer_input = timestep[0];
        
        // Process through each layer
        for (int layer = 0; layer < num_layers; layer++) {
            layer_input = lstm_layer_forward(layer_input, layer);
            if (is_training) {
                layer_input = apply_dropout(layer_input);
            }
        }
        
        // Final output through fully connected layer
        output = matrix_vector_mul(fc_weights, layer_input);
        output = elementwise_add(output, fc_bias);
    }
    
    return output;
}

// AdamState implementations
LSTMPredictor::AdamState::AdamState(const std::vector<std::vector<float>>& param_size) 
    : step(0),
      exp_avg(param_size.size(), std::vector<float>(param_size[0].size(), 0.0f)),
      exp_avg_sq(param_size.size(), std::vector<float>(param_size[0].size(), 0.0f)) {}

LSTMPredictor::AdamState::AdamState(const std::vector<float>& param_size) 
    : step(0),
      exp_avg(1, std::vector<float>(param_size.size(), 0.0f)),
      exp_avg_sq(1, std::vector<float>(param_size.size(), 0.0f)) {}

void LSTMPredictor::init_adam_optimizer(float learning_rate) {
    lr = learning_rate;
    
    // Initialize Adam states for layer weights and biases
    for (int layer = 0; layer < num_layers; ++layer) {
        // Input-hidden weights and biases
        adam_states["w_ih_" + std::to_string(layer)] = AdamState(w_ih[layer]);
        adam_states["b_ih_" + std::to_string(layer)] = AdamState(b_ih[layer]);
        
        // Hidden-hidden weights and biases
        adam_states["w_hh_" + std::to_string(layer)] = AdamState(w_hh[layer]);
        adam_states["b_hh_" + std::to_string(layer)] = AdamState(b_hh[layer]);
    }
    
    // Initialize Adam states for fully connected layer
    adam_states["fc_weights"] = AdamState(fc_weights);
    adam_states["fc_bias"] = AdamState(fc_bias);
}