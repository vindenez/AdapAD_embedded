#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <config.hpp>
#include <algorithm>

// Constructor to initialize with parameters in PyTorch order
LSTMPredictor::LSTMPredictor(int num_classes, int input_size, int hidden_size, int num_layers, int lookback_len)
    : input_size(input_size),
      hidden_size(hidden_size),
      output_size(num_classes),
      num_layers(num_layers),
      lookback_len(lookback_len),
      is_training(false) {

    // Initialize all storage vectors with proper sizes
    layer_inputs.resize(num_layers);
    layer_gates.resize(num_layers);
    layer_h_states.resize(num_layers);
    layer_c_states.resize(num_layers);
    layer_gradients.resize(num_layers);

    // Initialize states
    h_states.resize(num_layers, std::vector<float>(hidden_size, 0.0f));
    c_states.resize(num_layers, std::vector<float>(hidden_size, 0.0f));

    // Initialize layer storage
    layer_inputs.resize(num_layers);
    layer_gates.resize(num_layers);
    layer_h_states.resize(num_layers);
    layer_c_states.resize(num_layers);
    layer_gradients.resize(num_layers);

    // Initialize gradient storage
    dw_ih.resize(num_layers);
    dw_hh.resize(num_layers);
    db_ih.resize(num_layers);
    db_hh.resize(num_layers);

    // PyTorch uses k=1/sqrt(hidden_size) for the range [-k, k]
    float k = 1.0f / std::sqrt(hidden_size);
    std::uniform_real_distribution<float> weight_dist(-k, k);
    auto weight_init = [this, &weight_dist](float) { return weight_dist(gen); };

    // Initialize weights and biases vectors first
    w_ih.resize(num_layers);
    w_hh.resize(num_layers);
    b_ih.resize(num_layers);
    b_hh.resize(num_layers);

    for (int layer = 0; layer < num_layers; layer++) {
        // Initialize weights using same distribution for all gates
        if (layer == 0) {
            w_ih[layer] = init_weights(4 * hidden_size, input_size, weight_init);
        } else {
            w_ih[layer] = init_weights(4 * hidden_size, hidden_size, weight_init);
        }
        w_hh[layer] = init_weights(4 * hidden_size, hidden_size, weight_init);

        // Initialize biases with consistent forget gate bias
        b_ih[layer] = std::vector<float>(4 * hidden_size, 0.0f);
        b_hh[layer] = std::vector<float>(4 * hidden_size, 0.0f);
        
        // Set forget gate bias to 1.0 in both input and hidden biases
        for (size_t i = hidden_size; i < 2 * hidden_size; i++) {
            b_ih[layer][i] = 1.0f;  // Consistent with PyTorch default
            b_hh[layer][i] = 0.0f;  // Zero for hidden bias to avoid double counting
        }
    }

    // Even smaller initialization for FC layer
    float fc_k = 0.01f / std::sqrt(hidden_size);  // Reduce by factor of 100
    std::uniform_real_distribution<float> fc_dist(-fc_k, fc_k);
    auto fc_init = [this, &fc_dist](float) { return fc_dist(gen); };
    fc_weights = init_weights(output_size, hidden_size, fc_init);
    fc_bias = std::vector<float>(output_size, 0.0f);  // Initialize bias to 0

    // Initialize gradients
    dw_ih.resize(num_layers);
    dw_hh.resize(num_layers);
    db_ih.resize(num_layers);
    db_hh.resize(num_layers);

    // Initialize proper dimensions for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        size_t input_dim = (layer == 0) ? input_size : hidden_size;
        dw_ih[layer].resize(4 * hidden_size, std::vector<float>(input_dim, 0.0f));
        dw_hh[layer].resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        db_ih[layer].resize(4 * hidden_size, 0.0f);
        db_hh[layer].resize(4 * hidden_size, 0.0f);
    }
    
    // Zero FC gradients without clearing structure
    dw_fc.resize(output_size, std::vector<float>(hidden_size, 0.0f));
    db_fc.resize(output_size, 0.0f);
    
    // Clear layer gradients storage
    layer_gradients.clear();
    layer_gradients.resize(num_layers);
}


std::vector<std::vector<float>> LSTMPredictor::init_weights(int rows, int cols, 
                                                          const std::function<float(float)>& init_func) {
    std::vector<std::vector<float>> weights(rows, std::vector<float>(cols));
    float scale = std::sqrt(2.0f / (rows + cols));  // Xavier initialization
    for (auto& row : weights) {
        for (auto& val : row) {
            val = init_func(scale);  // Pass scale to init_func
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
    // Don't clear the structures, just zero the values
    for (int layer = 0; layer < num_layers; ++layer) {
        for (auto& row : dw_ih[layer]) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
        for (auto& row : dw_hh[layer]) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
        std::fill(db_ih[layer].begin(), db_ih[layer].end(), 0.0f);
        std::fill(db_hh[layer].begin(), db_hh[layer].end(), 0.0f);
    }
    
    // Zero FC gradients without clearing structure
    for (auto& row : dw_fc) {
        std::fill(row.begin(), row.end(), 0.0f);
    }
    std::fill(db_fc.begin(), db_fc.end(), 0.0f);
    
    // Clear and resize layer gradients
    layer_gradients.clear();
    layer_gradients.resize(num_layers);
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
        // Get the final prediction (from last timestep)
        const auto& final_output = outputs_list.back();
        
        // Compute MSE gradients with scaling
        const float initial_grad_scale = 1.0f;
        for (size_t i = 0; i < output_size; i++) {
            grad_output[i] = initial_grad_scale * (final_output[i] - targets[i]);
        }
    } else {
        throw std::runtime_error("Unsupported loss function: " + loss_function);
    }
    
    // After computing initial gradients
    std::cout << "Initial FC gradient magnitude: " 
              << std::accumulate(grad_output.begin(), grad_output.end(), 0.0f,
                               [](float a, float b) { return a + b * b; }) << std::endl;
    
    // Backward through fully connected layer
    auto dh = matrix_vector_mul_transpose(fc_weights, grad_output);
    
    // Update FC gradients before LSTM backward
    const auto& final_hidden = outputs_list.back();
    dw_fc = outer_product(grad_output, final_hidden);
    db_fc = grad_output;
    
    // Initialize layer gradients storage if needed
    layer_gradients.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        layer_gradients[layer].resize(layer_inputs[layer].size());
        for (auto& grad : layer_gradients[layer]) {
            grad.resize(hidden_size, 0.0f);
        }
    }
    
    // Backward through LSTM layers
    for (int layer = num_layers - 1; layer >= 0; --layer) {
        lstm_layer_backward(layer, dh);
        if (layer > 0) {
            // Get accumulated gradients from the current layer
            dh = std::vector<float>(hidden_size, 0.0f);
            for (const auto& timestep_grad : layer_gradients[layer - 1]) {
                dh = elementwise_add(dh, timestep_grad);
            }
        }
    }
}

void LSTMPredictor::lstm_layer_backward(int layer, const std::vector<float>& grad_output) {
    // Validate layer index first
    if (layer < 0 || layer >= num_layers) {
        throw std::runtime_error("Invalid layer index: " + std::to_string(layer));
    }
    
    // Ensure storage vectors are properly sized
    if (layer_inputs.size() <= layer || layer_inputs[layer].empty()) {
        std::cout << "Layer " << layer << " inputs size: " 
                  << (layer_inputs.size() <= layer ? 0 : layer_inputs[layer].size()) << std::endl;
        throw std::runtime_error("Layer inputs not properly initialized");
    }
    if (layer_gates.size() <= layer || layer_gates[layer].empty()) {
        throw std::runtime_error("Layer " + std::to_string(layer) + " gates not initialized");
    }
    if (layer_h_states.size() <= layer || layer_h_states[layer].empty()) {
        throw std::runtime_error("Layer " + std::to_string(layer) + " h_states not initialized");
    }
    if (layer_c_states.size() <= layer || layer_c_states[layer].empty()) {
        throw std::runtime_error("Layer " + std::to_string(layer) + " c_states not initialized");
    }
    
    // Add debug prints for input gradients
    std::cout << "Layer " << layer << " input gradient magnitude: "
              << std::accumulate(grad_output.begin(), grad_output.end(), 0.0f,
                               [](float a, float b) { return a + b * b; }) << std::endl;
    
    // Initialize gradient storage
    size_t seq_length = layer_inputs[layer].size();
    if (layer_gradients.size() <= layer) {
        layer_gradients.resize(layer + 1);
    }
    layer_gradients[layer].resize(seq_length);
    for (auto& grad : layer_gradients[layer]) {
        grad.resize(hidden_size, 0.0f);
    }
    
    // Initialize accumulated gradients with proper sizes
    std::vector<float> dh_next = grad_output;  // Copy initial gradient
    std::vector<float> dc_next(hidden_size, 0.0f);
    
    // Backward through time
    for (int t = seq_length - 1; t >= 0; --t) {
        // Get states for current timestep
        const auto& current_c = layer_c_states[layer][t];
        const auto& prev_c = (t > 0) ? layer_c_states[layer][t - 1] : 
                                      std::vector<float>(hidden_size, 0.0f);
        const auto& prev_h = (t > 0) ? layer_h_states[layer][t - 1] : 
                                      std::vector<float>(hidden_size, 0.0f);
        
        // Extract gates
        const auto& gates = layer_gates[layer][t];
        size_t gate_size = hidden_size;
        auto i_gate = std::vector<float>(gates.begin(), gates.begin() + gate_size);
        auto f_gate = std::vector<float>(gates.begin() + gate_size, gates.begin() + 2 * gate_size);
        auto g_gate = std::vector<float>(gates.begin() + 2 * gate_size, gates.begin() + 3 * gate_size);
        auto o_gate = std::vector<float>(gates.begin() + 3 * gate_size, gates.end());
        
        // Get input for this timestep
        const auto& input = (layer == 0) ? layer_inputs[layer][t] : layer_h_states[layer-1][t];
        
        // Compute gate gradients
        auto dh = dh_next;
        auto dc = dc_next;
        
        // Output gate gradient
        auto do_gate = elementwise_mul(dh, tanh_vector(current_c));
        do_gate = elementwise_mul(do_gate, sigmoid_derivative(o_gate));
        
        // Cell state gradient
        dc = elementwise_add(dc, elementwise_mul(dh, elementwise_mul(o_gate, dtanh_vector(current_c))));
        
        // Input and forget gate gradients
        auto di_gate = elementwise_mul(dc, g_gate);
        di_gate = elementwise_mul(di_gate, sigmoid_derivative(i_gate));
        
        auto df_gate = elementwise_mul(dc, prev_c);
        df_gate = elementwise_mul(df_gate, sigmoid_derivative(f_gate));
        
        // Cell input gradient
        auto dg_gate = elementwise_mul(dc, i_gate);
        dg_gate = elementwise_mul(dg_gate, tanh_derivative(g_gate));
        
        // Concatenate gate gradients
        auto dgate_concat = concatenate_vectors({di_gate, df_gate, dg_gate, do_gate});
        
        // Update weight gradients
        for (size_t i = 0; i < dgate_concat.size(); ++i) {
            for (size_t j = 0; j < input.size(); ++j) {
                dw_ih[layer][i][j] += dgate_concat[i] * input[j];
                if (t > 0) {
                    dw_hh[layer][i][j] += dgate_concat[i] * prev_h[j];
                }
            }
            db_ih[layer][i] += dgate_concat[i];
            db_hh[layer][i] += dgate_concat[i];
        }
        
        // Compute gradients for next timestep
        if (layer > 0) {
            auto input_grad = matrix_vector_mul(transpose(w_ih[layer]), dgate_concat);
            layer_gradients[layer - 1][t] = input_grad;
        }
        
        dh_next = matrix_vector_mul(transpose(w_hh[layer]), dgate_concat);
        dc_next = elementwise_mul(dc, f_gate);
    }
}

void LSTMPredictor::update_parameters_adam() {    
    // Hyperparameters matching PyTorch defaults
    const float lr = 0.001f;              // Learning rate (PyTorch default)
    const float beta1 = 0.9f;             // First moment decay (PyTorch default)
    const float beta2 = 0.999f;           // Second moment decay (PyTorch default)
    const float epsilon = 1e-8f;          // Numerical stability (PyTorch default)
    const float lstm_weight_decay = 0.0f; // Default weight decay
    const float fc_weight_decay = 0.0f;   // Default weight decay

    // Lambda for updating weights (2D)
    auto update_weights = [&](std::vector<std::vector<float>>& params,
                            const std::vector<std::vector<float>>& grad,
                            AdamState& state,
                            float current_weight_decay) {
        state.step += 1;
        
        // PyTorch-style bias correction terms
        float bias_correction1 = 1.0f - std::pow(beta1, state.step);
        float bias_correction2 = 1.0f - std::pow(beta2, state.step);
        
        for (size_t i = 0; i < params.size(); ++i) {
            for (size_t j = 0; j < params[i].size(); ++j) {
                float grad_val = grad[i][j];
                
                // Skip zero gradients for efficiency
                if (grad_val == 0.0f) continue;
                
                // Apply weight decay (L2 regularization)
                if (current_weight_decay > 0.0f) {
                    grad_val += current_weight_decay * params[i][j];
                }
                
                // Update first moment (momentum)
                state.exp_avg[i][j] = beta1 * state.exp_avg[i][j] + (1.0f - beta1) * grad_val;
                
                // Update second moment (RMSprop)
                state.exp_avg_sq[i][j] = beta2 * state.exp_avg_sq[i][j] + 
                                        (1.0f - beta2) * grad_val * grad_val;
                
                // Compute bias-corrected moments (PyTorch style)
                float exp_avg_corrected = state.exp_avg[i][j] / bias_correction1;
                float exp_avg_sq_corrected = state.exp_avg_sq[i][j] / bias_correction2;
                
                // Update parameters with numerical stability
                float denom = std::sqrt(exp_avg_sq_corrected) + epsilon;
                params[i][j] -= lr * exp_avg_corrected / denom;
            }
        }
    };

    // Lambda for updating biases (1D)
    auto update_biases = [&](std::vector<float>& params,
                           const std::vector<float>& grad,
                           AdamState& state,
                           float current_weight_decay) {
        state.step += 1;
        float bias_correction1 = 1.0f - std::pow(beta1, state.step);
        float bias_correction2 = 1.0f - std::pow(beta2, state.step);
        
        for (size_t i = 0; i < params.size(); ++i) {
            float grad_val = grad[i];
            
            // Apply weight decay if specified
            if (current_weight_decay > 0.0f) {
                grad_val += current_weight_decay * params[i];
            }
            
            // Update moments using PyTorch's approach
            state.exp_avg[0][i] = beta1 * state.exp_avg[0][i] + (1.0f - beta1) * grad_val;
            state.exp_avg_sq[0][i] = beta2 * state.exp_avg_sq[0][i] + 
                                    (1.0f - beta2) * grad_val * grad_val;
            
            // Compute bias-corrected moments
            float exp_avg_corrected = state.exp_avg[0][i] / bias_correction1;
            float exp_avg_sq_corrected = state.exp_avg_sq[0][i] / bias_correction2;
            
            // Update parameters
            float denom = std::sqrt(exp_avg_sq_corrected) + epsilon;
            params[i] -= lr * exp_avg_corrected / denom;
        }
    };
    
    // Update LSTM layers with lstm_weight_decay
    for (int layer = 0; layer < num_layers; ++layer) {
        update_weights(w_ih[layer], dw_ih[layer], 
                      adam_states["w_ih_" + std::to_string(layer)], 
                      lstm_weight_decay);
        update_weights(w_hh[layer], dw_hh[layer], 
                      adam_states["w_hh_" + std::to_string(layer)], 
                      lstm_weight_decay);
        update_biases(b_ih[layer], db_ih[layer], 
                     adam_states["b_ih_" + std::to_string(layer)], 
                     lstm_weight_decay);
        update_biases(b_hh[layer], db_hh[layer], 
                     adam_states["b_hh_" + std::to_string(layer)], 
                     lstm_weight_decay);
    }

    // Update with higher weight decay
    weight_decay = fc_weight_decay;
    update_weights(fc_weights, dw_fc, adam_states["fc_weights"], fc_weight_decay);
    update_biases(fc_bias, db_fc, adam_states["fc_bias"], fc_weight_decay);

    // Clip FC gradients
    float fc_grad_norm = 0.0f;
    for (const auto& row : dw_fc) {
        for (const auto& g : row) {
            fc_grad_norm += g * g;
        }
    }
    fc_grad_norm = std::sqrt(fc_grad_norm);
    
    if (fc_grad_norm > 5.0f) {  // Increased from 1.0f
        float scale = 5.0f / fc_grad_norm;  // Use the new threshold
        for (auto& row : dw_fc) {
            for (auto& g : row) {
                g *= scale;
            }
        }
    }
}

float LSTMPredictor::compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        std::cout << "MSE Loss Error - Output size: " << output.size() 
                  << ", Target size: " << target.size() << std::endl;
        throw std::runtime_error("Output and target size mismatch in MSE loss calculation");
    }
    
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = output[i] - target[i];
        loss += diff * diff;
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
    // Store input for this layer if training
    if (is_training) {
        // Ensure the vector exists and has proper size
        if (layer_inputs.size() <= layer) {
            layer_inputs.resize(layer + 1);
        }
        layer_inputs[layer].push_back(x);
    }
    
    // Validate layer index first
    if (layer < 0 || layer >= num_layers) {
        throw std::runtime_error("Invalid layer index in forward pass");
    }

    // Initialize storage vectors if needed
    if (layer_inputs.size() <= layer) {
        layer_inputs.resize(layer + 1);
    }
    if (layer_gates.size() <= layer) {
        layer_gates.resize(layer + 1);
    }
    if (layer_h_states.size() <= layer) {
        layer_h_states.resize(layer + 1);
    }
    if (layer_c_states.size() <= layer) {
        layer_c_states.resize(layer + 1);
    }

    // Initialize states if empty
    if (h_states[layer].empty()) {
        h_states[layer].resize(hidden_size, 0.0f);
    }
    if (c_states[layer].empty()) {
        c_states[layer].resize(hidden_size, 0.0f);
    }

    // Get previous states
    const auto& prev_h = h_states[layer];
    const auto& prev_c = c_states[layer];

    // Validate state dimensions
    if (prev_h.size() != hidden_size || prev_c.size() != hidden_size) {
        throw std::runtime_error("State size mismatch in lstm_layer_forward");
    }

    // Compute gates pre-activations
    auto gates_ih = matrix_vector_mul(w_ih[layer], x);
    auto gates_hh = matrix_vector_mul(w_hh[layer], prev_h);
    
    // Add biases with proper forget gate handling
    auto gates_pre = elementwise_add(elementwise_add(gates_ih, gates_hh),
                                   elementwise_add(b_ih[layer], b_hh[layer]));
    
    // No need for additional bias here since it's already in b_ih
    // Remove lines 676-680 that add extra biases
    
    // Split and activate gates
    int chunk_size = hidden_size;
    auto i_gate = sigmoid_vector(std::vector<float>(gates_pre.begin(), 
                                                  gates_pre.begin() + chunk_size));
    auto f_gate = sigmoid_vector(std::vector<float>(gates_pre.begin() + chunk_size, 
                                                  gates_pre.begin() + 2 * chunk_size));
    auto g_gate = tanh_vector(std::vector<float>(gates_pre.begin() + 2 * chunk_size, 
                                                  gates_pre.begin() + 3 * chunk_size));
    auto o_gate = sigmoid_vector(std::vector<float>(gates_pre.begin() + 3 * chunk_size, 
                                                  gates_pre.end()));

    // Update cell state with bounds checking
    std::vector<float> new_c(hidden_size);
    std::vector<float> new_h(hidden_size);
    
    // Consistent scaling factors (around line 693-694)
    const float state_scale = 5.0f;     // Cell state clipping threshold
    const float gate_scale = 1.0f;      // Gate activation scale
    
    // Update cell state with consistent scaling
    for (size_t i = 0; i < hidden_size; i++) {
        // Scale gate outputs consistently
        float scaled_forget = f_gate[i] * gate_scale;
        float scaled_input = i_gate[i] * gate_scale;
        float scaled_cell = g_gate[i] * gate_scale;
        
        new_c[i] = scaled_forget * prev_c[i] + scaled_input * scaled_cell;
        
        // Consistent state clipping
        if (std::abs(new_c[i]) > state_scale) {
            new_c[i] = state_scale * tanh_func(new_c[i] / state_scale);
        }
    }
    
    // Consistent hidden state update
    for (size_t i = 0; i < hidden_size; i++) {
        new_h[i] = o_gate[i] * gate_scale * tanh_func(new_c[i]);
    }

    // Store states for backward pass
    if (is_training) {
        // Ensure vectors exist and have proper size
        if (layer_h_states.size() <= layer) {
            layer_h_states.resize(layer + 1);
        }
        if (layer_c_states.size() <= layer) {
            layer_c_states.resize(layer + 1);
        }
        if (layer_gates.size() <= layer) {
            layer_gates.resize(layer + 1);
        }

        // Store h and c states
        layer_h_states[layer].push_back(new_h);
        layer_c_states[layer].push_back(new_c);
        
        // Store gates
        std::vector<float> gates_concat;
        gates_concat.insert(gates_concat.end(), i_gate.begin(), i_gate.end());
        gates_concat.insert(gates_concat.end(), f_gate.begin(), f_gate.end());
        gates_concat.insert(gates_concat.end(), g_gate.begin(), g_gate.end());
        gates_concat.insert(gates_concat.end(), o_gate.begin(), o_gate.end());
        layer_gates[layer].push_back(gates_concat);
    }

    // Update states
    h_states[layer] = new_h;
    c_states[layer] = new_c;

    return new_h;
}

std::vector<float> LSTMPredictor::forward(const std::vector<std::vector<std::vector<float>>>& input) {
    if (is_training) {
        // Clear all storage vectors but maintain their structure
        layer_inputs.resize(num_layers);
        layer_gates.resize(num_layers);
        layer_h_states.resize(num_layers);
        layer_c_states.resize(num_layers);
        layer_gradients.resize(num_layers);
        outputs_list.clear();
        
        // Initialize each layer's storage
        for (int layer = 0; layer < num_layers; ++layer) {
            layer_inputs[layer].clear();
            layer_gates[layer].resize(1);  // We know we'll have one timestep
            layer_h_states[layer].resize(1);  // Initialize with one timestep
            layer_c_states[layer].resize(1);  // Initialize with one timestep
            
            // Pre-allocate storage for gates and states
            layer_gates[layer][0].resize(4 * hidden_size, 0.0f);
            layer_h_states[layer][0].resize(hidden_size, 0.0f);
            layer_c_states[layer][0].resize(hidden_size, 0.0f);
        }
    }

    // Process input through LSTM layers
    std::vector<float> layer_input = input[0][0];  // Get the input vector
    
    // Store initial input for layer 0
    if (is_training) {
        layer_inputs[0].push_back(layer_input);
    }
    
    // Process through LSTM layers
    for (int layer = 0; layer < num_layers; layer++) {
        layer_input = lstm_layer_forward(layer_input, layer);
        
        // Store input for next layer if training
        if (is_training && layer < num_layers - 1) {
            layer_inputs[layer + 1].push_back(layer_input);
        }
    }
    
    // Final fully connected layer
    auto output = matrix_vector_mul(fc_weights, layer_input);
    output = elementwise_add(output, fc_bias);
    
    // Validate output size
    if (output.size() != output_size) {
        std::cout << "FC output size: " << output.size() 
                  << ", Expected output size: " << output_size << std::endl;
        throw std::runtime_error("FC layer output size mismatch");
    }
    
    // Store output if training
    if (is_training) {
        outputs_list.push_back(output);
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

void LSTMPredictor::init_adam_optimizer() {
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

std::vector<float> LSTMPredictor::get_state() const {
    std::vector<float> state;
    
    // Helper to flatten 2D vector into 1D
    auto flatten = [&state](const std::vector<std::vector<float>>& matrix) {
        for (const auto& row : matrix) {
            state.insert(state.end(), row.begin(), row.end());
        }
    };
    
    // Save layer weights and biases
    for (int layer = 0; layer < num_layers; ++layer) {
        flatten(w_ih[layer]);
        flatten(w_hh[layer]);
        state.insert(state.end(), b_ih[layer].begin(), b_ih[layer].end());
        state.insert(state.end(), b_hh[layer].begin(), b_hh[layer].end());
    }
    
    // Save fully connected layer
    flatten(fc_weights);
    state.insert(state.end(), fc_bias.begin(), fc_bias.end());
    
    return state;
}

void LSTMPredictor::load_state(const std::vector<float>& state) {
    size_t pos = 0;
    
    // Helper to unflatten 1D vector into 2D
    auto unflatten = [&pos, &state](std::vector<std::vector<float>>& matrix) {
        for (auto& row : matrix) {
            for (auto& val : row) {
                val = state[pos++];
            }
        }
    };
    
    // Load layer weights and biases
    for (int layer = 0; layer < num_layers; ++layer) {
        unflatten(w_ih[layer]);
        unflatten(w_hh[layer]);
        
        for (auto& val : b_ih[layer]) {
            val = state[pos++];
        }
        for (auto& val : b_hh[layer]) {
            val = state[pos++];
        }
    }
    
    // Load fully connected layer
    unflatten(fc_weights);
    for (auto& val : fc_bias) {
        val = state[pos++];
    }
    
    // Reset optimizer states
    init_adam_optimizer();
}

// Add helper function for gradient clipping
void LSTMPredictor::clip_gradients(std::vector<std::vector<float>>& gradients, float max_norm) {
    float grad_norm = 0.0f;
    for (const auto& row : gradients) {
        for (const auto& g : row) {
            grad_norm += g * g;
        }
    }
    grad_norm = std::sqrt(grad_norm);
    
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        for (auto& row : gradients) {
            for (auto& g : row) {
                g *= scale;
            }
        }
    }
}

void LSTMPredictor::reset_states() {
    // Reset hidden and cell states
    for (int layer = 0; layer < num_layers; ++layer) {
        std::fill(h_states[layer].begin(), h_states[layer].end(), 0.0f);
        std::fill(c_states[layer].begin(), c_states[layer].end(), 0.0f);
    }
    
    // Clear storage vectors
    layer_inputs.clear();
    layer_gates.clear();
    layer_h_states.clear();
    layer_c_states.clear();
    
    // Reinitialize storage vectors
    layer_inputs.resize(num_layers);
    layer_gates.resize(num_layers);
    layer_h_states.resize(num_layers);
    layer_c_states.resize(num_layers);
}

void LSTMPredictor::reshape_input(const std::vector<float>& input_sequence, 
                                      std::vector<std::vector<std::vector<float>>>& reshaped) {
    // Validate input size
    if (input_sequence.size() < lookback_len) {
        throw std::runtime_error("Input sequence length is less than lookback_len");
    }

    // Clear and resize with proper dimensions
    reshaped.clear();
    reshaped.resize(1);  // batch_size = 1
    reshaped[0].resize(1);  // sequence_length = 1 (matching PyTorch reshape)
    reshaped[0][0].resize(lookback_len);  // input_size = lookback_len
    
    // Copy the last lookback_len elements
    for (int i = 0; i < lookback_len; i++) {
        reshaped[0][0][i] = input_sequence[input_sequence.size() - lookback_len + i];
    }
}

bool LSTMPredictor::check_gradients(const std::vector<std::vector<std::vector<float>>>& input,
                                  const std::vector<float>& target,
                                  float epsilon,
                                  float threshold) {
    // Use a larger threshold for numerical stability
    if (threshold == 0.0f) {
        threshold = 1e-5;
    }
    epsilon = std::max(epsilon, 1e-7f);  // Ensure epsilon isn't too small
    
    // Validate input and target sizes
    if (target.size() != output_size) {
        std::cout << "Target size: " << target.size() 
                  << ", Expected output size: " << output_size << std::endl;
        throw std::runtime_error("Target size does not match model output size");
    }
    
    bool was_training = is_training;
    is_training = true;
    
    // First forward and backward pass to compute analytical gradients
    zero_grad();
    auto output = forward(input);
    
    // Validate output size
    if (output.size() != target.size()) {
        std::cout << "Forward output size: " << output.size() 
                  << ", Target size: " << target.size() << std::endl;
        throw std::runtime_error("Forward pass output size mismatch");
    }
    
    backward(target, "mse");
    
    // Store analytical gradients
    std::vector<std::vector<std::vector<float>>> stored_dw_ih = dw_ih;
    std::vector<std::vector<std::vector<float>>> stored_dw_hh = dw_hh;
    std::vector<std::vector<float>> stored_db_ih = db_ih;
    std::vector<std::vector<float>> stored_db_hh = db_hh;
    std::vector<std::vector<float>> stored_dw_fc = dw_fc;
    std::vector<float> stored_db_fc = db_fc;
    
    bool gradients_ok = true;
    
    // Helper function to check gradients
    auto check_param_gradients = [&](const std::string& param_name,
                                   std::vector<std::vector<float>>& param,
                                   std::vector<std::vector<float>>& grad,
                                   int layer) {
        for (size_t i = 0; i < param.size(); ++i) {
            for (size_t j = 0; j < param[i].size(); ++j) {
                auto numerical_grad = compute_numerical_gradient(input, target, param, i, j, epsilon);
                float rel_error = std::abs(grad[i][j] - numerical_grad[0]) / 
                                (std::abs(grad[i][j]) + std::abs(numerical_grad[0]) + epsilon);
                
                if (rel_error > threshold) {
                    std::cout << "Gradient check failed for " << param_name 
                             << "[" << layer << "][" << i << "][" << j << "]" << std::endl;
                    std::cout << "Analytical: " << grad[i][j] 
                             << " Numerical: " << numerical_grad[0] 
                             << " Relative Error: " << rel_error << std::endl;
                    gradients_ok = false;
                }
            }
        }
    };

    // Check LSTM layer gradients
    for (int layer = 0; layer < num_layers; ++layer) {
        // Check input-hidden weights
        check_param_gradients("w_ih", w_ih[layer], dw_ih[layer], layer);
        
        // Check hidden-hidden weights
        check_param_gradients("w_hh", w_hh[layer], dw_hh[layer], layer);
        
        // Check input-hidden biases
        for (size_t i = 0; i < b_ih[layer].size(); ++i) {
            float original_val = b_ih[layer][i];
            
            // Compute loss with b_ih + epsilon
            b_ih[layer][i] = original_val + epsilon;
            auto output_plus = forward(input);
            float loss_plus = compute_mse_loss(output_plus, target);
            
            // Compute loss with b_ih - epsilon
            b_ih[layer][i] = original_val - epsilon;
            auto output_minus = forward(input);
            float loss_minus = compute_mse_loss(output_minus, target);
            
            // Restore original value
            b_ih[layer][i] = original_val;
            
            float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);
            float rel_error = std::abs(db_ih[layer][i] - numerical_grad) / 
                            (std::abs(db_ih[layer][i]) + std::abs(numerical_grad) + epsilon);
            
            if (rel_error > threshold) {
                std::cout << "Gradient check failed for b_ih[" << layer << "][" << i << "]" << std::endl;
                std::cout << "Analytical: " << db_ih[layer][i] 
                         << " Numerical: " << numerical_grad 
                         << " Relative Error: " << rel_error << std::endl;
                gradients_ok = false;
            }
        }
        
        // Check hidden-hidden biases
        for (size_t i = 0; i < b_hh[layer].size(); ++i) {
            float original_val = b_hh[layer][i];
            
            // Compute loss with b_hh + epsilon
            b_hh[layer][i] = original_val + epsilon;
            auto output_plus = forward(input);
            float loss_plus = compute_mse_loss(output_plus, target);
            
            // Compute loss with b_hh - epsilon
            b_hh[layer][i] = original_val - epsilon;
            auto output_minus = forward(input);
            float loss_minus = compute_mse_loss(output_minus, target);
            
            // Restore original value
            b_hh[layer][i] = original_val;
            
            float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);
            float rel_error = std::abs(db_hh[layer][i] - numerical_grad) / 
                            (std::abs(db_hh[layer][i]) + std::abs(numerical_grad) + epsilon);
            
            if (rel_error > threshold) {
                std::cout << "Gradient check failed for b_hh[" << layer << "][" << i << "]" << std::endl;
                std::cout << "Analytical: " << db_hh[layer][i] 
                         << " Numerical: " << numerical_grad 
                         << " Relative Error: " << rel_error << std::endl;
                gradients_ok = false;
            }
        }
    }
    
    // Check FC layer gradients
    check_param_gradients("fc_weights", fc_weights, dw_fc, 0);
    
    // Check FC bias gradients
    for (size_t i = 0; i < fc_bias.size(); ++i) {
        float original_val = fc_bias[i];
        
        // Compute loss with fc_bias + epsilon
        fc_bias[i] = original_val + epsilon;
        auto output_plus = forward(input);
        float loss_plus = compute_mse_loss(output_plus, target);
        
        // Compute loss with fc_bias - epsilon
        fc_bias[i] = original_val - epsilon;
        auto output_minus = forward(input);
        float loss_minus = compute_mse_loss(output_minus, target);
        
        // Restore original value
        fc_bias[i] = original_val;
        
        float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);
        float rel_error = std::abs(db_fc[i] - numerical_grad) / 
                        (std::abs(db_fc[i]) + std::abs(numerical_grad) + epsilon);
        
        if (rel_error > threshold) {
            std::cout << "Gradient check failed for fc_bias[" << i << "]" << std::endl;
            std::cout << "Analytical: " << db_fc[i] 
                     << " Numerical: " << numerical_grad 
                     << " Relative Error: " << rel_error << std::endl;
            gradients_ok = false;
        }
    }
    
    is_training = was_training;
    return gradients_ok;
}

std::vector<float> LSTMPredictor::compute_numerical_gradient(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<float>& target,
    std::vector<std::vector<float>>& param,
    size_t i, size_t j,
    float epsilon) {
    
    // Store original parameter value
    float original_val = param[i][j];
    
    // Compute loss with param + epsilon
    param[i][j] = original_val + epsilon;
    auto output_plus = forward(input);
    float loss_plus = compute_mse_loss(output_plus, target);
    
    // Compute loss with param - epsilon
    param[i][j] = original_val - epsilon;
    auto output_minus = forward(input);
    float loss_minus = compute_mse_loss(output_minus, target);
    
    // Restore original parameter value
    param[i][j] = original_val;
    
    // Compute numerical gradient
    float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);
    return {numerical_grad};
}

float LSTMPredictor::vector_magnitude(const std::vector<float>& vec) {
    return std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0f,
                                   [](float a, float b) { return a + b * b; }));
}
