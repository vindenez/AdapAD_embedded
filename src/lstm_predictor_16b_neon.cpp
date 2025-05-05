#include "lstm_predictor_16b_neon.hpp"
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <arm_neon.h>
#include <execinfo.h>
#include <cxxabi.h> 

LSTMPredictor16bNEON::LSTMPredictor16bNEON(int num_classes, int input_size, int hidden_size, 
                            int num_layers, int lookback_len, 
                            bool batch_first)
    : num_classes(num_classes),
      num_layers(num_layers),
      input_size(input_size),
      hidden_size(hidden_size),
      seq_length(lookback_len),
      batch_first(batch_first),
      training_mode(false),
      online_learning_mode(false),
      is_cache_initialized(false) {
    
    std::cout << "Initializing 16-bit LSTM Predictor with:" << std::endl;
    std::cout << "- num_classes: " << num_classes << std::endl;
    std::cout << "- input_size: " << input_size << std::endl;
    std::cout << "- hidden_size: " << hidden_size << std::endl;
    std::cout << "- num_layers: " << num_layers << std::endl;
    std::cout << "- lookback_len: " << lookback_len << std::endl;
    
    // Pre-allocate LSTM layers
    lstm_layers.resize(num_layers);
    last_gradients.resize(num_layers);
    
    // Initialize velocity terms
    velocity_weight_ih.resize(num_layers);
    velocity_weight_hh.resize(num_layers);
    velocity_bias_ih.resize(num_layers);
    velocity_bias_hh.resize(num_layers);
    
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        velocity_weight_ih[layer].resize(4 * hidden_size, std::vector<float16_t>(input_size_layer, 0.0f16));
        velocity_weight_hh[layer].resize(4 * hidden_size, std::vector<float16_t>(hidden_size, 0.0f16));
        velocity_bias_ih[layer].resize(4 * hidden_size, 0.0f16);
        velocity_bias_hh[layer].resize(4 * hidden_size, 0.0f16);
    }
    
    velocity_fc_weight.resize(num_classes, std::vector<float16_t>(hidden_size, 0.0f16));
    velocity_fc_bias.resize(num_classes, 0.0f16);
    
    // Pre-allocate layer cache
    layer_cache.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        layer_cache[layer].resize(1);  // One batch
        layer_cache[layer][0].resize(lookback_len);  
        
        for (auto& seq : layer_cache[layer][0]) {
            int expected_input_size = (layer == 0) ? input_size : hidden_size;
            seq.input.resize(expected_input_size, 0.0f16);
            seq.prev_hidden.resize(hidden_size, 0.0f16);
            seq.prev_cell.resize(hidden_size, 0.0f16);
            seq.cell_state.resize(hidden_size, 0.0f16);
            seq.input_gate.resize(hidden_size, 0.0f16);
            seq.forget_gate.resize(hidden_size, 0.0f16);
            seq.cell_candidate.resize(hidden_size, 0.0f16);
            seq.output_gate.resize(hidden_size, 0.0f16);
            seq.hidden_state.resize(hidden_size, 0.0f16);
        }
    }
    
    // Pre-allocate gradients
    for (auto& grad : last_gradients) {
        int input_size_layer = (current_layer == 0) ? input_size : hidden_size;
        grad.weight_ih_grad.resize(4 * hidden_size, std::vector<float16_t>(input_size_layer, 0.0f16));
        grad.weight_hh_grad.resize(4 * hidden_size, std::vector<float16_t>(hidden_size, 0.0f16));
        grad.bias_ih_grad.resize(4 * hidden_size, 0.0f16);
        grad.bias_hh_grad.resize(4 * hidden_size, 0.0f16);
    }
    
    // Pre-allocate states
    h_state.resize(num_layers, std::vector<float16_t>(hidden_size, 0.0f16));
    c_state.resize(num_layers, std::vector<float16_t>(hidden_size, 0.0f16));
    
    // Initialize weights
    initialize_weights();
    reset_states();
    
    std::cout << "16-bit LSTM Predictor initialization complete" << std::endl;
}

void LSTMPredictor16bNEON::reset_states() {
    c_state.clear();
    h_state.clear();
    c_state.resize(num_layers, std::vector<float16_t>(hidden_size, 0.0f16));
    h_state.resize(num_layers, std::vector<float16_t>(hidden_size, 0.0f16));
}

float16x8_t LSTMPredictor16bNEON::sigmoid_neon_fp16(float16x8_t x) {
    // Constants
    float16x8_t v_one = vdupq_n_f16(1.0f16);
    float16x8_t v_min = vdupq_n_f16(-10.0f16);
    float16x8_t v_max = vdupq_n_f16(10.0f16);

    // Clamp input
    x = vmaxq_f16(x, v_min);
    x = vminq_f16(x, v_max);

    // Fast sigmoid approximation for fp16
    float16x8_t v_neg_x = vnegq_f16(x);
    
    // Improved polynomial approximation for exp(-x) in fp16
    float16x8_t v_exp_approx = vmlaq_n_f16(v_one, v_neg_x, 0.4f16);
    float16x8_t x_sq = vmulq_f16(v_neg_x, v_neg_x);
    v_exp_approx = vmlaq_n_f16(v_exp_approx, x_sq, 0.13f16);
    
    return vdivq_f16(v_one, vaddq_f16(v_one, v_exp_approx));
}

float16x8_t LSTMPredictor16bNEON::tanh_neon_fp16(float16x8_t x) {
    float16x8_t v_one = vdupq_n_f16(1.0f16);
    float16x8_t v_min = vdupq_n_f16(-10.0f16);
    float16x8_t v_max = vdupq_n_f16(10.0f16);

    // Clamp input
    x = vmaxq_f16(x, v_min);
    x = vminq_f16(x, v_max);

    // Fast tanh approximation using polynomial for fp16
    float16x8_t x_sq = vmulq_f16(x, x);
    float16x8_t x_cube = vmulq_f16(x, x_sq);
    
    // Pade approximation: tanh(x) ≈ x / (1 + x² / 3)
    float16x8_t numerator = x;
    float16x8_t denominator = vmlaq_n_f16(v_one, x_sq, 0.333f16);
    
    return vdivq_f16(numerator, denominator);
}

std::vector<float16_t> LSTMPredictor16bNEON::lstm_cell_forward(
    const std::vector<float16_t>& input,
    std::vector<float16_t>& h_state,
    std::vector<float16_t>& c_state,
    const LSTMLayer16bit& layer) {

    // Verify dimensions
    int expected_layer_input = (current_layer == 0) ? input_size : hidden_size;
    if (input.size() != expected_layer_input) {
        throw std::runtime_error("Input size mismatch in lstm_cell_forward_16bit");
    }
    
    // Resize state vectors if needed
    if (h_state.size() != hidden_size) h_state.resize(hidden_size, 0.0f16);
    if (c_state.size() != hidden_size) c_state.resize(hidden_size, 0.0f16);

    // Cache entry for training mode
    LSTMCacheEntry16bit* cache_entry = nullptr;
    if (training_mode) {
        if (current_layer >= layer_cache.size() ||
            current_batch >= layer_cache[current_layer].size() ||
            current_timestep >= layer_cache[current_layer][current_batch].size()) {
            throw std::runtime_error("Invalid cache access");
        }
        
        cache_entry = &layer_cache[current_layer][current_batch][current_timestep];
        
        // Resize cache vectors
        cache_entry->input.resize(expected_layer_input);
        cache_entry->prev_hidden.resize(hidden_size);
        cache_entry->prev_cell.resize(hidden_size);
        cache_entry->cell_state.resize(hidden_size);
        cache_entry->input_gate.resize(hidden_size);
        cache_entry->forget_gate.resize(hidden_size);
        cache_entry->cell_candidate.resize(hidden_size);
        cache_entry->output_gate.resize(hidden_size);
        cache_entry->hidden_state.resize(hidden_size);
        
        // Cache input and previous states
        std::copy(input.begin(), input.end(), cache_entry->input.begin());
        std::copy(h_state.begin(), h_state.end(), cache_entry->prev_hidden.begin());
        std::copy(c_state.begin(), c_state.end(), cache_entry->prev_cell.begin());
    }
    
    // Initialize gates with biases
    std::vector<float16_t> gates(4 * hidden_size);

    // Add biases using NEON
    for (int h = 0; h + 7 < hidden_size; h += 8) {
        // Load bias vectors
        float16x8_t v_bias_ih_i = vld1q_f16(&layer.bias_ih[h]);
        float16x8_t v_bias_hh_i = vld1q_f16(&layer.bias_hh[h]);
        float16x8_t v_bias_ih_f = vld1q_f16(&layer.bias_ih[hidden_size + h]);
        float16x8_t v_bias_hh_f = vld1q_f16(&layer.bias_hh[hidden_size + h]);
        float16x8_t v_bias_ih_g = vld1q_f16(&layer.bias_ih[2 * hidden_size + h]);
        float16x8_t v_bias_hh_g = vld1q_f16(&layer.bias_hh[2 * hidden_size + h]);
        float16x8_t v_bias_ih_o = vld1q_f16(&layer.bias_ih[3 * hidden_size + h]);
        float16x8_t v_bias_hh_o = vld1q_f16(&layer.bias_hh[3 * hidden_size + h]);

        // Add corresponding biases
        float16x8_t v_gate_i = vaddq_f16(v_bias_ih_i, v_bias_hh_i);
        float16x8_t v_gate_f = vaddq_f16(v_bias_ih_f, v_bias_hh_f);
        float16x8_t v_gate_g = vaddq_f16(v_bias_ih_g, v_bias_hh_g);
        float16x8_t v_gate_o = vaddq_f16(v_bias_ih_o, v_bias_hh_o);

        // Store in gates vector
        vst1q_f16(&gates[h], v_gate_i);
        vst1q_f16(&gates[hidden_size + h], v_gate_f);
        vst1q_f16(&gates[2 * hidden_size + h], v_gate_g);
        vst1q_f16(&gates[3 * hidden_size + h], v_gate_o);
    }

    // Handle remaining elements
    for (int h = (hidden_size / 8) * 8; h < hidden_size; ++h) {
        gates[h] = layer.bias_ih[h] + layer.bias_hh[h];
        gates[hidden_size + h] = layer.bias_ih[hidden_size + h] + layer.bias_hh[hidden_size + h];
        gates[2 * hidden_size + h] = layer.bias_ih[2 * hidden_size + h] + layer.bias_hh[2 * hidden_size + h];
        gates[3 * hidden_size + h] = layer.bias_ih[3 * hidden_size + h] + layer.bias_hh[3 * hidden_size + h];
    }
    
    // Input to hidden contribution using optimized NEON fp16 matrix multiplication
    for (size_t i = 0; i < input.size(); ++i) {
        float16x8_t v_input = vdupq_n_f16(input[i]);

        for (int gate_type = 0; gate_type < 4; ++gate_type) {
            int gate_offset = gate_type * hidden_size;

            for (int h = 0; h + 7 < hidden_size; h += 8) {
                float16x8_t v_weights = vld1q_f16(&layer.weight_ih[gate_offset + h][i]);
                float16x8_t v_gates = vld1q_f16(&gates[gate_offset + h]);
                v_gates = vfmaq_f16(v_gates, v_weights, v_input);
                vst1q_f16(&gates[gate_offset + h], v_gates);
            }

            for (int h = (hidden_size / 8) * 8; h < hidden_size; ++h) {
                gates[gate_offset + h] += layer.weight_ih[gate_offset + h][i] * input[i];
            }
        }
    }
    
    // Hidden to hidden contribution using NEON
    for (size_t i = 0; i < hidden_size; ++i) {
        float16x8_t v_h_state = vdupq_n_f16(h_state[i]);

        for (int gate_type = 0; gate_type < 4; ++gate_type) {
            int gate_offset = gate_type * hidden_size;

            for (int h = 0; h + 7 < hidden_size; h += 8) {
                float16x8_t v_weights = vld1q_f16(&layer.weight_hh[gate_offset + h][i]);
                float16x8_t v_gates = vld1q_f16(&gates[gate_offset + h]);
                v_gates = vfmaq_f16(v_gates, v_weights, v_h_state);
                vst1q_f16(&gates[gate_offset + h], v_gates);
            }

            for (int h = (hidden_size / 8) * 8; h < hidden_size; ++h) {
                gates[gate_offset + h] += layer.weight_hh[gate_offset + h][i] * h_state[i];
            }
        }
    }

    // Apply activations and update states
    std::vector<float16_t> output(hidden_size);

    // Process 8 elements at a time
    for (int h = 0; h + 7 < hidden_size; h += 8) {
        // Load gate values
        float16x8_t v_gate_i = vld1q_f16(&gates[h]);
        float16x8_t v_gate_f = vld1q_f16(&gates[hidden_size + h]);
        float16x8_t v_gate_g = vld1q_f16(&gates[2 * hidden_size + h]);
        float16x8_t v_gate_o = vld1q_f16(&gates[3 * hidden_size + h]);

        // Apply activations
        float16x8_t v_i_t = sigmoid_neon_fp16(v_gate_i);
        float16x8_t v_f_t = sigmoid_neon_fp16(v_gate_f);
        float16x8_t v_cell_candidate = tanh_neon_fp16(v_gate_g);
        float16x8_t v_o_t = sigmoid_neon_fp16(v_gate_o);

        // Load cell state
        float16x8_t v_c_state = vld1q_f16(&c_state[h]);

        // Compute new cell state: c_t = f_t * c_{t-1} + i_t * g_t
        float16x8_t v_fc = vmulq_f16(v_f_t, v_c_state);
        float16x8_t v_ig = vmulq_f16(v_i_t, v_cell_candidate);
        float16x8_t v_new_c = vaddq_f16(v_fc, v_ig);

        // Compute tanh of new cell state
        float16x8_t v_tanh_c = tanh_neon_fp16(v_new_c);

        // Compute new hidden state: h_t = o_t * tanh(c_t)
        float16x8_t v_new_h = vmulq_f16(v_o_t, v_tanh_c);

        // Store new states
        vst1q_f16(&c_state[h], v_new_c);
        vst1q_f16(&h_state[h], v_new_h);
        vst1q_f16(&output[h], v_new_h);

        // Store in cache if in training mode
        if (training_mode && cache_entry) {
            vst1q_f16(&cache_entry->input_gate[h], v_i_t);
            vst1q_f16(&cache_entry->forget_gate[h], v_f_t);
            vst1q_f16(&cache_entry->cell_candidate[h], v_cell_candidate);
            vst1q_f16(&cache_entry->output_gate[h], v_o_t);
            vst1q_f16(&cache_entry->cell_state[h], v_new_c);
            vst1q_f16(&cache_entry->hidden_state[h], v_new_h);
        }
    }

    // Handle remaining elements
    for (int h = (hidden_size / 8) * 8; h < hidden_size; ++h) {
        // Use fp16 arithmetic throughout
        float16_t i_t = 1.0f16 / (1.0f16 + exp((float16_t)(-gates[h])));
        float16_t f_t = 1.0f16 / (1.0f16 + exp((float16_t)(-gates[hidden_size + h])));
        float16_t cell_candidate = tanh((float16_t)gates[2 * hidden_size + h]);
        float16_t o_t = 1.0f16 / (1.0f16 + exp((float16_t)(-gates[3 * hidden_size + h])));

        // Update cell state
        float16_t new_cell = f_t * c_state[h] + i_t * cell_candidate;
        c_state[h] = new_cell;
        
        // Update hidden state
        float16_t new_hidden = o_t * tanh(new_cell);
        h_state[h] = new_hidden;
        output[h] = new_hidden;

        // Store in cache if in training mode
        if (training_mode && cache_entry) {
            cache_entry->input_gate[h] = i_t;
            cache_entry->forget_gate[h] = f_t;
            cache_entry->cell_candidate[h] = cell_candidate;
            cache_entry->output_gate[h] = o_t;
            cache_entry->cell_state[h] = new_cell;
            cache_entry->hidden_state[h] = new_hidden;
        }
    }
    
    return output;
}

void LSTMPredictor16bNEON::backward_linear_layer(
    const std::vector<float16_t>& grad_output,
    const std::vector<float16_t>& last_hidden,
    std::vector<std::vector<float16_t>>& weight_grad,
    std::vector<float16_t>& bias_grad,
    std::vector<float16_t>& input_grad) {
    
    if (grad_output.size() != num_classes) {
        throw std::invalid_argument(
            "grad_output size mismatch: " + std::to_string(grad_output.size()) +
            " != " + std::to_string(num_classes)
        );
    }
    
    if (last_hidden.size() != hidden_size) {
        throw std::invalid_argument(
            "last_hidden size mismatch: " + std::to_string(last_hidden.size()) +
            " != " + std::to_string(hidden_size)
        );
    }
    
    weight_grad.resize(num_classes, std::vector<float16_t>(hidden_size, 0.0f16));
    bias_grad = grad_output;
    input_grad.resize(hidden_size, 0.0f16);
    
    // Compute weight gradients using NEON fp16
    for (int i = 0; i < num_classes; ++i) {
        float16x8_t v_grad_output_i = vdupq_n_f16(grad_output[i]);
        
        int j = 0;
        for (; j + 7 < hidden_size; j += 8) {
            float16x8_t v_last_hidden = vld1q_f16(&last_hidden[j]);
            float16x8_t v_result = vmulq_f16(v_grad_output_i, v_last_hidden);
            vst1q_f16(&weight_grad[i][j], v_result);
        }
        
        for (; j < hidden_size; ++j) {
            weight_grad[i][j] = grad_output[i] * last_hidden[j];
        }
    }
    
    // Compute input gradients
    int i = 0;
    float16x8_t v_zeros = vdupq_n_f16(0.0f16);
    for (; i + 7 < hidden_size; i += 8) {
        vst1q_f16(&input_grad[i], v_zeros);
    }
    for (; i < hidden_size; ++i) {
        input_grad[i] = 0.0f16;
    }
    
    for (int j = 0; j < num_classes; ++j) {
        float16x8_t v_grad_j = vdupq_n_f16(grad_output[j]);
        
        i = 0;
        for (; i + 7 < hidden_size; i += 8) {
            float16x8_t v_input_grad = vld1q_f16(&input_grad[i]);
            float16x8_t v_weights = vld1q_f16(&fc_weight[j][i]);
            v_input_grad = vfmaq_f16(v_input_grad, v_weights, v_grad_j);
            vst1q_f16(&input_grad[i], v_input_grad);
        }
        
        for (; i < hidden_size; ++i) {
            input_grad[i] += fc_weight[j][i] * grad_output[j];
        }
    }
}

std::vector<LSTMPredictor16bNEON::LSTMGradients16bit> LSTMPredictor16bNEON::backward_lstm_layer(
    const std::vector<float16_t>& grad_output,
    const std::vector<std::vector<std::vector<LSTMCacheEntry16bit>>>& cache,
    float16_t learning_rate) {
    
    if (grad_output.size() != hidden_size) {
        throw std::runtime_error("grad_output size mismatch in backward_lstm_layer_16bit");
    }
    
    if (cache.size() != num_layers) {
        throw std::runtime_error("cache layer count mismatch in backward_lstm_layer_16bit");
    }
    
    std::vector<LSTMGradients16bit> layer_grads(num_layers);
    
    // Initialize gradients for each layer
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        layer_grads[layer].weight_ih_grad.resize(4 * hidden_size, std::vector<float16_t>(input_size_layer, 0.0f16));
        layer_grads[layer].weight_hh_grad.resize(4 * hidden_size, std::vector<float16_t>(hidden_size, 0.0f16));
        layer_grads[layer].bias_ih_grad.resize(4 * hidden_size, 0.0f16);
        layer_grads[layer].bias_hh_grad.resize(4 * hidden_size, 0.0f16);
    }
    
    // Initialize dh_next and dc_next for each layer
    std::vector<std::vector<float16_t>> dh_next(num_layers, std::vector<float16_t>(hidden_size, 0.0f16));
    std::vector<std::vector<float16_t>> dc_next(num_layers, std::vector<float16_t>(hidden_size, 0.0f16));
    
    // Start from the last layer and move backward
    for (int layer = num_layers - 1; layer >= 0; --layer) {
        std::vector<float16_t> dh = dh_next[layer];
        std::vector<float16_t> dc = dc_next[layer];
        
        if (current_batch >= cache[layer].size()) {
            throw std::runtime_error("Cache batch index out of bounds");
        }
        
        const auto& layer_cache = cache[layer][current_batch];
        
        // If this is the last layer, add grad output
        if (layer == num_layers - 1) {
            for (int h = 0; h + 7 < hidden_size; h += 8) {
                float16x8_t v_dh = vld1q_f16(&dh[h]);
                float16x8_t v_grad = vld1q_f16(&grad_output[h]);
                v_dh = vaddq_f16(v_dh, v_grad);
                vst1q_f16(&dh[h], v_dh);
            }
            
            for (int h = (hidden_size / 8) * 8; h < hidden_size; ++h) {
                dh[h] += grad_output[h];
            }
        }

        // Process each time step in reverse order
        for (int t = layer_cache.size() - 1; t >= 0; --t) {
            const auto& cache_entry = layer_cache[t];
            
            std::vector<float16_t> dh_prev(hidden_size, 0.0f16);
            std::vector<float16_t> dc_prev(hidden_size, 0.0f16);

            // Process each hidden unit using NEON fp16
            for (int h = 0; h + 7 < hidden_size; h += 8) {
                // Constants
                float16x8_t v_one = vdupq_n_f16(1.0f16);

                // Load cell state and compute tanh and its derivative
                float16x8_t v_cell_state = vld1q_f16(&cache_entry.cell_state[h]);
                float16x8_t v_tanh_c = tanh_neon_fp16(v_cell_state);
                float16x8_t v_tanh_c_squared = vmulq_f16(v_tanh_c, v_tanh_c);
                float16x8_t v_dtanh_c = vsubq_f16(v_one, v_tanh_c_squared);

                // Load dh and compute dc_t
                float16x8_t v_dh = vld1q_f16(&dh[h]);
                float16x8_t v_output_gate = vld1q_f16(&cache_entry.output_gate[h]);
                float16x8_t v_dc = vld1q_f16(&dc[h]);

                // Compute dc_t = dh * o * (1 - tanh²(c)) + dc
                float16x8_t v_dho_dtanh = vmulq_f16(v_dh, v_output_gate);
                v_dho_dtanh = vmulq_f16(v_dho_dtanh, v_dtanh_c);
                float16x8_t v_dc_t = vaddq_f16(v_dho_dtanh, v_dc);

                // Load gates
                float16x8_t v_input_gate = vld1q_f16(&cache_entry.input_gate[h]);
                float16x8_t v_forget_gate = vld1q_f16(&cache_entry.forget_gate[h]);
                float16x8_t v_cell_candidate = vld1q_f16(&cache_entry.cell_candidate[h]);
                float16x8_t v_prev_cell = vld1q_f16(&cache_entry.prev_cell[h]);

                // Compute gate gradients using NEON
                // Output gate: do = dh * tanh(c) * o * (1 - o)
                float16x8_t v_one_minus_o = vsubq_f16(v_one, v_output_gate);
                float16x8_t v_do_t = vmulq_f16(v_dh, v_tanh_c);
                v_do_t = vmulq_f16(v_do_t, vmulq_f16(v_output_gate, v_one_minus_o));

                // Input gate: di = dc * g * i * (1 - i)
                float16x8_t v_one_minus_i = vsubq_f16(v_one, v_input_gate);
                float16x8_t v_di_t = vmulq_f16(v_dc_t, v_cell_candidate);
                v_di_t = vmulq_f16(v_di_t, vmulq_f16(v_input_gate, v_one_minus_i));

                // Forget gate: df = dc * c_prev * f * (1 - f)
                float16x8_t v_one_minus_f = vsubq_f16(v_one, v_forget_gate);
                float16x8_t v_df_t = vmulq_f16(v_dc_t, v_prev_cell);
                v_df_t = vmulq_f16(v_df_t, vmulq_f16(v_forget_gate, v_one_minus_f));

                // Cell candidate: dg = dc * i * (1 - g²)
                float16x8_t v_g_squared = vmulq_f16(v_cell_candidate, v_cell_candidate);
                float16x8_t v_one_minus_g_squared = vsubq_f16(v_one, v_g_squared);
                float16x8_t v_dg_t = vmulq_f16(v_dc_t, vmulq_f16(v_input_gate, v_one_minus_g_squared));

                // Accumulate bias gradients
                for (int idx = 0; idx < 8; ++idx) {
                    layer_grads[layer].bias_ih_grad[h + idx] += vgetq_lane_f16(v_di_t, idx);
                    layer_grads[layer].bias_ih_grad[hidden_size + h + idx] += vgetq_lane_f16(v_df_t, idx);
                    layer_grads[layer].bias_ih_grad[2 * hidden_size + h + idx] += vgetq_lane_f16(v_dg_t, idx);
                    layer_grads[layer].bias_ih_grad[3 * hidden_size + h + idx] += vgetq_lane_f16(v_do_t, idx);

                    // Compute dc_prev
                    dc_prev[h + idx] = vgetq_lane_f16(v_dc_t, idx) * cache_entry.forget_gate[h + idx];
                }
            }

            // Handle remaining elements
            for (int h = (hidden_size / 8) * 8; h < hidden_size; ++h) {
                float16_t tanh_c = tanh(cache_entry.cell_state[h]);
                float16_t dho = dh[h];
                
                // Cell state gradient
                float16_t dc_t = dho * cache_entry.output_gate[h] * (1.0f16 - tanh_c * tanh_c);
                dc_t += dc[h];

                // Gate gradients
                float16_t do_t = dho * tanh_c * cache_entry.output_gate[h] * (1.0f16 - cache_entry.output_gate[h]);
                float16_t di_t = dc_t * cache_entry.cell_candidate[h] * cache_entry.input_gate[h] * (1.0f16 - cache_entry.input_gate[h]);
                float16_t df_t = dc_t * cache_entry.prev_cell[h] * cache_entry.forget_gate[h] * (1.0f16 - cache_entry.forget_gate[h]);
                float16_t dg_t = dc_t * cache_entry.input_gate[h] * (1.0f16 - cache_entry.cell_candidate[h] * cache_entry.cell_candidate[h]);

                // Accumulate bias gradients
                layer_grads[layer].bias_ih_grad[h] += di_t;
                layer_grads[layer].bias_ih_grad[hidden_size + h] += df_t;
                layer_grads[layer].bias_ih_grad[2 * hidden_size + h] += dg_t;
                layer_grads[layer].bias_ih_grad[3 * hidden_size + h] += do_t;

                // Cell state gradient for previous timestep
                dc_prev[h] = dc_t * cache_entry.forget_gate[h];
            }

            // Calculate weight gradients using NEON when possible
            int input_size_layer = (layer == 0) ? input_size : hidden_size;

            for (int h = 0; h < hidden_size; ++h) {
                float16_t tanh_c = tanh(cache_entry.cell_state[h]);
                float16_t dho = dh[h];
                
                // Recompute gradients for weight calculations
                float16_t dc_t = dho * cache_entry.output_gate[h] * (1.0f16 - tanh_c * tanh_c) + dc[h];
                float16_t do_t = dho * tanh_c * cache_entry.output_gate[h] * (1.0f16 - cache_entry.output_gate[h]);
                float16_t di_t = dc_t * cache_entry.cell_candidate[h] * cache_entry.input_gate[h] * (1.0f16 - cache_entry.input_gate[h]);
                float16_t df_t = dc_t * cache_entry.prev_cell[h] * cache_entry.forget_gate[h] * (1.0f16 - cache_entry.forget_gate[h]);
                float16_t dg_t = dc_t * cache_entry.input_gate[h] * (1.0f16 - cache_entry.cell_candidate[h] * cache_entry.cell_candidate[h]);
                
                // Accumulate weight gradients using NEON
                float16x8_t v_di_t = vdupq_n_f16(di_t);
                float16x8_t v_df_t = vdupq_n_f16(df_t);
                float16x8_t v_dg_t = vdupq_n_f16(dg_t);
                float16x8_t v_do_t = vdupq_n_f16(do_t);
                
                int j = 0;
                for (; j + 7 < input_size_layer; j += 8) {
                    float16x8_t v_input = vld1q_f16(&cache_entry.input[j]);
                    
                    float16x8_t v_weight_ih_grad_i = vld1q_f16(&layer_grads[layer].weight_ih_grad[h][j]);
                    float16x8_t v_weight_ih_grad_f = vld1q_f16(&layer_grads[layer].weight_ih_grad[hidden_size + h][j]);
                    float16x8_t v_weight_ih_grad_g = vld1q_f16(&layer_grads[layer].weight_ih_grad[2 * hidden_size + h][j]);
                    float16x8_t v_weight_ih_grad_o = vld1q_f16(&layer_grads[layer].weight_ih_grad[3 * hidden_size + h][j]);
                    
                    v_weight_ih_grad_i = vfmaq_f16(v_weight_ih_grad_i, v_di_t, v_input);
                    v_weight_ih_grad_f = vfmaq_f16(v_weight_ih_grad_f, v_df_t, v_input);
                    v_weight_ih_grad_g = vfmaq_f16(v_weight_ih_grad_g, v_dg_t, v_input);
                    v_weight_ih_grad_o = vfmaq_f16(v_weight_ih_grad_o, v_do_t, v_input);
                    
                    vst1q_f16(&layer_grads[layer].weight_ih_grad[h][j], v_weight_ih_grad_i);
                    vst1q_f16(&layer_grads[layer].weight_ih_grad[hidden_size + h][j], v_weight_ih_grad_f);
                    vst1q_f16(&layer_grads[layer].weight_ih_grad[2 * hidden_size + h][j], v_weight_ih_grad_g);
                    vst1q_f16(&layer_grads[layer].weight_ih_grad[3 * hidden_size + h][j], v_weight_ih_grad_o);
                }
                
                for (; j < input_size_layer; ++j) {
                    float16_t input_j = cache_entry.input[j];
                    layer_grads[layer].weight_ih_grad[h][j] += di_t * input_j;
                    layer_grads[layer].weight_ih_grad[hidden_size + h][j] += df_t * input_j;
                    layer_grads[layer].weight_ih_grad[2 * hidden_size + h][j] += dg_t * input_j;
                    layer_grads[layer].weight_ih_grad[3 * hidden_size + h][j] += do_t * input_j;
                }
                
                // Accumulate hidden-hidden weight gradients and dh_prev
                j = 0;
                for (; j + 7 < hidden_size; j += 8) {
                    float16x8_t v_h_prev = vld1q_f16(&cache_entry.prev_hidden[j]);
                    
                    float16x8_t v_weight_hh_grad_i = vld1q_f16(&layer_grads[layer].weight_hh_grad[h][j]);
                    float16x8_t v_weight_hh_grad_f = vld1q_f16(&layer_grads[layer].weight_hh_grad[hidden_size + h][j]);
                    float16x8_t v_weight_hh_grad_g = vld1q_f16(&layer_grads[layer].weight_hh_grad[2 * hidden_size + h][j]);
                    float16x8_t v_weight_hh_grad_o = vld1q_f16(&layer_grads[layer].weight_hh_grad[3 * hidden_size + h][j]);
                    
                    v_weight_hh_grad_i = vfmaq_f16(v_weight_hh_grad_i, v_di_t, v_h_prev);
                    v_weight_hh_grad_f = vfmaq_f16(v_weight_hh_grad_f, v_df_t, v_h_prev);
                    v_weight_hh_grad_g = vfmaq_f16(v_weight_hh_grad_g, v_dg_t, v_h_prev);
                    v_weight_hh_grad_o = vfmaq_f16(v_weight_hh_grad_o, v_do_t, v_h_prev);
                    
                    vst1q_f16(&layer_grads[layer].weight_hh_grad[h][j], v_weight_hh_grad_i);
                    vst1q_f16(&layer_grads[layer].weight_hh_grad[hidden_size + h][j], v_weight_hh_grad_f);
                    vst1q_f16(&layer_grads[layer].weight_hh_grad[2 * hidden_size + h][j], v_weight_hh_grad_g);
                    vst1q_f16(&layer_grads[layer].weight_hh_grad[3 * hidden_size + h][j], v_weight_hh_grad_o);
                    
                    // Accumulate dh_prev
                    float16x8_t v_dh_prev = vld1q_f16(&dh_prev[j]);
                    float16x8_t v_weight_i = vld1q_f16(&lstm_layers[layer].weight_hh[h][j]);
                    float16x8_t v_weight_f = vld1q_f16(&lstm_layers[layer].weight_hh[hidden_size + h][j]);
                    float16x8_t v_weight_g = vld1q_f16(&lstm_layers[layer].weight_hh[2 * hidden_size + h][j]);
                    float16x8_t v_weight_o = vld1q_f16(&lstm_layers[layer].weight_hh[3 * hidden_size + h][j]);
                    
                    v_dh_prev = vfmaq_f16(v_dh_prev, v_di_t, v_weight_i);
                    v_dh_prev = vfmaq_f16(v_dh_prev, v_df_t, v_weight_f);
                    v_dh_prev = vfmaq_f16(v_dh_prev, v_dg_t, v_weight_g);
                    v_dh_prev = vfmaq_f16(v_dh_prev, v_do_t, v_weight_o);
                    
                    vst1q_f16(&dh_prev[j], v_dh_prev);
                }
                
                for (; j < hidden_size; ++j) {
                    float16_t h_prev_j = cache_entry.prev_hidden[j];
                    layer_grads[layer].weight_hh_grad[h][j] += di_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[hidden_size + h][j] += df_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[2 * hidden_size + h][j] += dg_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[3 * hidden_size + h][j] += do_t * h_prev_j;
                    
                    // Accumulate gradients for next timestep's hidden state
                    dh_prev[j] += di_t * lstm_layers[layer].weight_hh[h][j];
                    dh_prev[j] += df_t * lstm_layers[layer].weight_hh[hidden_size + h][j];
                    dh_prev[j] += dg_t * lstm_layers[layer].weight_hh[2 * hidden_size + h][j];
                    dh_prev[j] += do_t * lstm_layers[layer].weight_hh[3 * hidden_size + h][j];
                }
            }
            
            // Update gradients for next timestep
            dh = dh_prev;
            dc = dc_prev;
        }
        
        // Pass gradients to next layer
        if (layer > 0) {
            dh_next[layer - 1] = dh;
            dc_next[layer - 1] = dc;
        }
    }
    
    last_gradients = layer_grads;
    return layer_grads;
}

void LSTMPredictor16bNEON::apply_sgd_update(
    std::vector<std::vector<float16_t>>& weights,
    std::vector<std::vector<float16_t>>& grads,
    float16_t learning_rate,
    float16_t momentum) {

    // Find which velocity terms to use based on dimensions
    std::vector<std::vector<float16_t>>* velocity_terms = nullptr;
    if (weights.size() == num_classes && weights[0].size() == hidden_size) {
        velocity_terms = &velocity_fc_weight;
    } else if (weights.size() == 4 * hidden_size) {
        if (weights[0].size() == input_size) {
            velocity_terms = &velocity_weight_ih[current_layer];
        } else if (weights[0].size() == hidden_size) {
            velocity_terms = &velocity_weight_hh[current_layer];
        }
    }

    if (!velocity_terms) {
        throw std::runtime_error("Unknown weight dimensions in apply_sgd_update_16bit");
    }

    // NEON constants for vectorized operations
    float16x8_t v_momentum = vdupq_n_f16(momentum);  
    float16x8_t v_neg_lr = vdupq_n_f16(-learning_rate);  
    float16x8_t v_clip_max = vdupq_n_f16(1.0f16); 
    float16x8_t v_clip_min = vdupq_n_f16(-1.0f16); 

    // Process each row of weights
    for (size_t i = 0; i < weights.size(); ++i) {
        size_t j = 0;

        // Process 8 elements at once using fp16 arithmetic
        for (; j + 7 < weights[i].size(); j += 8) {
            // Load gradients and velocity
            float16x8_t v_grad = vld1q_f16(&grads[i][j]);
            float16x8_t v_velocity = vld1q_f16(&(*velocity_terms)[i][j]);

            // Clip gradients between -1.0 and 1.0
            v_grad = vminq_f16(v_grad, v_clip_max);
            v_grad = vmaxq_f16(v_grad, v_clip_min);

            // Update velocity: v = momentum * v - lr * grad
            float16x8_t v_scaled_grad = vmulq_f16(v_grad, v_neg_lr);
            float16x8_t v_momentum_vel = vmulq_f16(v_velocity, v_momentum);
            v_velocity = vaddq_f16(v_momentum_vel, v_scaled_grad);

            // Store updated velocity
            vst1q_f16(&(*velocity_terms)[i][j], v_velocity);

            // Load weights
            float16x8_t v_weight = vld1q_f16(&weights[i][j]);

            // Update weights: w = w + v
            v_weight = vaddq_f16(v_weight, v_velocity);

            // Store updated weights
            vst1q_f16(&weights[i][j], v_weight);
        }

        // Handle remaining elements
        for (; j < weights[i].size(); ++j) {
            float16_t grad = grads[i][j];

            // Gradient clipping
            grad = std::max(std::min(grad, 1.0f16), -1.0f16);

            // Update velocity with momentum
            float16_t velocity = momentum * (*velocity_terms)[i][j] - learning_rate * grad;
            (*velocity_terms)[i][j] = velocity;

            // Update weights using velocity
            weights[i][j] += velocity;
        }
    }
}

void LSTMPredictor16bNEON::apply_sgd_update(
    std::vector<float16_t>& biases,
    std::vector<float16_t>& grads,
    float16_t learning_rate,
    float16_t momentum) {

    // Find which velocity terms to use
    std::vector<float16_t>* velocity_terms = nullptr;

    if (biases.size() == num_classes) {
        velocity_terms = &velocity_fc_bias;
    }
    else if (biases.size() == 4 * hidden_size) {
        for (int layer = 0; layer < num_layers; ++layer) {
            if (&biases == &lstm_layers[layer].bias_ih) {
                velocity_terms = &velocity_bias_ih[layer];
                current_layer = layer;
                break;
            } else if (&biases == &lstm_layers[layer].bias_hh) {
                velocity_terms = &velocity_bias_hh[layer];
                current_layer = layer;
                break;
            }
        }
    }

    if (!velocity_terms) {
        throw std::runtime_error("Unknown bias dimensions in apply_sgd_update_16bit");
    }

    // NEON constants
    float16x8_t v_momentum = vdupq_n_f16(momentum);
    float16x8_t v_neg_lr = vdupq_n_f16(-learning_rate);
    float16x8_t v_clip_max = vdupq_n_f16(1.0f16);
    float16x8_t v_clip_min = vdupq_n_f16(-1.0f16);

    size_t i = 0;

    // Process 8 elements at a time
    for (; i + 7 < biases.size(); i += 8) {
        // Load gradients and velocity
        float16x8_t v_grad = vld1q_f16(&grads[i]);
        float16x8_t v_velocity = vld1q_f16(&(*velocity_terms)[i]);

        // Clip gradients
        v_grad = vminq_f16(v_grad, v_clip_max);
        v_grad = vmaxq_f16(v_grad, v_clip_min);

        // Compute new velocity
        float16x8_t v_scaled_grad = vmulq_f16(v_grad, v_neg_lr);
        float16x8_t v_momentum_vel = vmulq_f16(v_velocity, v_momentum);
        v_velocity = vaddq_f16(v_momentum_vel, v_scaled_grad);

        // Store updated velocity
        vst1q_f16(&(*velocity_terms)[i], v_velocity);

        // Load biases
        float16x8_t v_bias = vld1q_f16(&biases[i]);

        // Update biases
        v_bias = vaddq_f16(v_bias, v_velocity);

        // Store updated biases
        vst1q_f16(&biases[i], v_bias);
    }

    // Handle remaining elements
    for (; i < biases.size(); ++i) {
        float16_t grad = grads[i];

        // Gradient clipping
        grad = std::max(std::min(grad, 1.0f16), -1.0f16);

        // Update velocity with momentum
        float16_t velocity = momentum * (*velocity_terms)[i] - learning_rate * grad;
        (*velocity_terms)[i] = velocity;

        // Update biases using velocity
        biases[i] += velocity;
    }
}

std::vector<float16_t> LSTMPredictor16bNEON::compute_mse_loss_gradient(
    const std::vector<float16_t>& output,
    const std::vector<float16_t>& target) {

    if (output.size() != target.size()) {
        throw std::runtime_error("Output and target size mismatch in compute_mse_loss_gradient");
    }

    std::vector<float16_t> gradient(output.size());

    size_t i = 0;
    // Process 8 elements at a time using fp16
    for (; i + 7 < output.size(); i += 8) {
        // Load output and target vectors
        float16x8_t v_output = vld1q_f16(&output[i]);
        float16x8_t v_target = vld1q_f16(&target[i]);

        // Calculate gradient: output - target
        float16x8_t v_gradient = vsubq_f16(v_output, v_target);

        // Store result
        vst1q_f16(&gradient[i], v_gradient);
    }

    // Handle remaining elements
    for (; i < output.size(); ++i) {
        gradient[i] = output[i] - target[i];
    }

    return gradient;
}

std::vector<float16_t> LSTMPredictor16bNEON::get_final_prediction(const LSTMOutput16bit& lstm_output) {
    std::vector<float16_t> final_output(num_classes, 0.0f16);

    // Get final hidden state
    const auto& final_hidden = lstm_output.sequence_output.back().back();

    // For each output class
    for (int i = 0; i < num_classes; ++i) {
        float16_t sum = fc_bias[i];
        int j = 0;
        float16x8_t v_sum = vdupq_n_f16(0.0f16);

        // Process 8 hidden units at a time using fp16
        for (; j + 7 < hidden_size; j += 8) {
            // Load weights and hidden state
            float16x8_t v_weight = vld1q_f16(&fc_weight[i][j]);
            float16x8_t v_hidden = vld1q_f16(&final_hidden[j]);

            // Multiply-accumulate
            v_sum = vfmaq_f16(v_sum, v_weight, v_hidden);
        }

        // Reduce vector sum to scalar
        float16x4_t v_sum_low = vget_low_f16(v_sum);
        float16x4_t v_sum_high = vget_high_f16(v_sum);
        v_sum_low = vadd_f16(v_sum_low, v_sum_high);
        
        // Further reduction
        float16x4_t v_sum_pair = vpadd_f16(v_sum_low, v_sum_low);
        float16_t scalar_sum = vget_lane_f16(vpadd_f16(v_sum_pair, v_sum_pair), 0);
        
        sum += scalar_sum;

        // Handle remaining elements
        for (; j < hidden_size; ++j) {
            sum += fc_weight[i][j] * final_hidden[j];
        }

        final_output[i] = sum;
    }

    return final_output;
}

void LSTMPredictor16bNEON::train_step(
    const std::vector<std::vector<std::vector<float16_t>>>& x,
    const std::vector<float16_t>& target,
    const LSTMOutput16bit& lstm_output,
    float16_t learning_rate) {

    try {
        // Validate input dimensions
        for (size_t batch = 0; batch < x.size(); ++batch) {
            for (size_t seq = 0; seq < x[batch].size(); ++seq) {
                if (x[batch][seq].size() != input_size) {
                    throw std::runtime_error(
                        "Input sequence dimension mismatch in train_step: batch " +
                        std::to_string(batch) + ", seq " + std::to_string(seq));
                }
            }
        }

        if (x.empty() || x[0].empty() || x[0][0].empty()) {
            throw std::runtime_error("Empty input tensor");
        }
        if (x[0][0].size() != input_size) {
            throw std::runtime_error("Input feature size mismatch");
        }
        if (target.size() != num_classes) {
            throw std::invalid_argument("Target size mismatch");
        }

        // 1. Get the final prediction
        auto output = get_final_prediction(lstm_output);

        // 2. Compute gradient of loss w.r.t. output
        auto grad_output = compute_mse_loss_gradient(output, target);

        // 3. Get final hidden state
        const auto& last_hidden = lstm_output.sequence_output.back().back();

        // 4. Backward pass through FC layer
        std::vector<std::vector<float16_t>> fc_weight_grad;
        std::vector<float16_t> fc_bias_grad;
        std::vector<float16_t> lstm_grad;
        backward_linear_layer(grad_output, last_hidden, fc_weight_grad, fc_bias_grad, lstm_grad);

        // 5. Update FC layer weights
        apply_sgd_update(fc_weight, fc_weight_grad, learning_rate, 0.9f16);
        apply_sgd_update(fc_bias, fc_bias_grad, learning_rate, 0.9f16);

        // 6. Backward pass through LSTM layers
        auto lstm_gradients = backward_lstm_layer(lstm_grad, layer_cache, learning_rate);

        // 7. Update LSTM layer parameters
        for (int layer = 0; layer < num_layers; ++layer) {
            if (layer >= lstm_layers.size()) {
                throw std::runtime_error("Layer index out of bounds: " + std::to_string(layer));
            }
            if (layer >= lstm_gradients.size()) {
                throw std::runtime_error("Gradient index out of bounds: " + std::to_string(layer));
            }

            // Update weights and biases
            apply_sgd_update(lstm_layers[layer].weight_ih, lstm_gradients[layer].weight_ih_grad, learning_rate, 0.9f16);
            apply_sgd_update(lstm_layers[layer].weight_hh, lstm_gradients[layer].weight_hh_grad, learning_rate, 0.9f16);
            apply_sgd_update(lstm_layers[layer].bias_ih, lstm_gradients[layer].bias_ih_grad, learning_rate, 0.9f16);
            apply_sgd_update(lstm_layers[layer].bias_hh, lstm_gradients[layer].bias_hh_grad, learning_rate, 0.9f16);
        }

        // 8. Clear temporary state
        clear_update_state();

    } catch (const std::exception& e) {
        clear_update_state();
        throw;
    }
}

void LSTMPredictor16bNEON::initialize_weights() {
    // Initialize with PyTorch's default initialization, then convert to fp16
    float k = 1.0f / std::sqrt(hidden_size);
    std::uniform_real_distribution<float> dist(-k, k);
    std::mt19937 gen(random_seed);

    // Initialize FC layer first
    fc_weight.resize(num_classes, std::vector<float16_t>(hidden_size));
    fc_bias.resize(num_classes);
    
    // Initialize FC weights and bias
    for (int i = 0; i < num_classes; ++i) {
        fc_bias[i] = 0.0f16;  // PyTorch default
        for (int j = 0; j < hidden_size; ++j) {
            fc_weight[i][j] = static_cast<float16_t>(dist(gen));
        }
    }

    // Initialize LSTM layers
    lstm_layers.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        
        // Initialize with PyTorch dimensions
        lstm_layers[layer].weight_ih.resize(4 * hidden_size, 
            std::vector<float16_t>(input_size_layer));
        lstm_layers[layer].weight_hh.resize(4 * hidden_size, 
            std::vector<float16_t>(hidden_size));
        lstm_layers[layer].bias_ih.resize(4 * hidden_size);
        lstm_layers[layer].bias_hh.resize(4 * hidden_size);
        
        // Initialize weights and biases
        for (int i = 0; i < 4 * hidden_size; ++i) {
            for (int j = 0; j < input_size_layer; ++j) {
                lstm_layers[layer].weight_ih[i][j] = static_cast<float16_t>(dist(gen));
            }
            for (int j = 0; j < hidden_size; ++j) {
                lstm_layers[layer].weight_hh[i][j] = static_cast<float16_t>(dist(gen));
            }
            lstm_layers[layer].bias_ih[i] = static_cast<float16_t>(dist(gen));
            lstm_layers[layer].bias_hh[i] = static_cast<float16_t>(dist(gen));
        }
    }
}

// Conversion functions for loading from fp32 models
void LSTMPredictor16bNEON::set_lstm_weights(int layer, 
                                         const std::vector<std::vector<float>>& w_ih,
                                         const std::vector<std::vector<float>>& w_hh) {
    if (layer < num_layers) {
        lstm_layers[layer].weight_ih.resize(w_ih.size(), std::vector<float16_t>(w_ih[0].size()));
        lstm_layers[layer].weight_hh.resize(w_hh.size(), std::vector<float16_t>(w_hh[0].size()));
        
        for (size_t i = 0; i < w_ih.size(); ++i) {
            for (size_t j = 0; j < w_ih[i].size(); ++j) {
                lstm_layers[layer].weight_ih[i][j] = static_cast<float16_t>(w_ih[i][j]);
            }
        }
        for (size_t i = 0; i < w_hh.size(); ++i) {
            for (size_t j = 0; j < w_hh[i].size(); ++j) {
                lstm_layers[layer].weight_hh[i][j] = static_cast<float16_t>(w_hh[i][j]);
            }
        }
    }
}

void LSTMPredictor16bNEON::set_lstm_bias(int layer,
                                      const std::vector<float>& b_ih,
                                      const std::vector<float>& b_hh) {
    if (layer < num_layers) {
        lstm_layers[layer].bias_ih.resize(b_ih.size());
        lstm_layers[layer].bias_hh.resize(b_hh.size());
        
        for (size_t i = 0; i < b_ih.size(); ++i) {
            lstm_layers[layer].bias_ih[i] = static_cast<float16_t>(b_ih[i]);
        }
        for (size_t i = 0; i < b_hh.size(); ++i) {
            lstm_layers[layer].bias_hh[i] = static_cast<float16_t>(b_hh[i]);
        }
    }
}

void LSTMPredictor16bNEON::set_fc_weights(const std::vector<std::vector<float>>& weights,
                                       const std::vector<float>& bias) {
    fc_weight.resize(weights.size(), std::vector<float16_t>(weights[0].size()));
    fc_bias.resize(bias.size());
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            fc_weight[i][j] = static_cast<float16_t>(weights[i][j]);
        }
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        fc_bias[i] = static_cast<float16_t>(bias[i]);
    }
}