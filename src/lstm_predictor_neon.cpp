#include "lstm_predictor_neon.hpp"
#include "config.hpp"
#include "matrix_utils.hpp"

#include <algorithm>
#include <cxxabi.h>
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <random>

#if USE_NEON
#include <arm_neon.h>
#endif

float32x4_t LSTMPredictorNEON::sigmoid_neon(float32x4_t x) {
    // Constants
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_min = vdupq_n_f32(-10.0f); 
    float32x4_t v_max = vdupq_n_f32(10.0f);

    // Clamp input to avoid overflow/underflow
    x = vmaxq_f32(x, v_min);
    x = vminq_f32(x, v_max);

    float32x4_t v_neg_x = vnegq_f32(x);

    // Calculate each element individually
    float exp_values[4];
    exp_values[0] = std::exp(vgetq_lane_f32(v_neg_x, 0));
    exp_values[1] = std::exp(vgetq_lane_f32(v_neg_x, 1));
    exp_values[2] = std::exp(vgetq_lane_f32(v_neg_x, 2));
    exp_values[3] = std::exp(vgetq_lane_f32(v_neg_x, 3));
    float32x4_t v_exp_neg_x = vld1q_f32(exp_values);

    // Sigmoid: 1 / (1 + exp(-x))
    float32x4_t v_denom = vaddq_f32(v_one, v_exp_neg_x);
    float32x4_t v_result = vdivq_f32(v_one, v_denom);

    return v_result;
}

float32x4_t LSTMPredictorNEON::tanh_neon(float32x4_t x) {
    float32x4_t v_min = vdupq_n_f32(-10.0f); 
    float32x4_t v_max = vdupq_n_f32(10.0f);

    // Clamp input to avoid overflow/underflow
    x = vmaxq_f32(x, v_min);
    x = vminq_f32(x, v_max);

    // Calculate each element individually
    float tanh_values[4];
    tanh_values[0] = std::tanh(vgetq_lane_f32(x, 0));
    tanh_values[1] = std::tanh(vgetq_lane_f32(x, 1));
    tanh_values[2] = std::tanh(vgetq_lane_f32(x, 2));
    tanh_values[3] = std::tanh(vgetq_lane_f32(x, 3));

    float32x4_t v_tanh_values = vld1q_f32(tanh_values);

    return v_tanh_values;
}

LSTMPredictorNEON::LSTMPredictorNEON(int num_classes, int input_size, int hidden_size,
                                     int num_layers, int lookback_len, bool batch_first)
    : LSTMPredictor(num_classes, input_size, hidden_size, num_layers, lookback_len, batch_first) {

    std::cout << "Initializing LSTM Predictor with:" << std::endl;
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
        velocity_weight_ih[layer].resize(4 * hidden_size,
                                         std::vector<float>(input_size_layer, 0.0f));
        velocity_weight_hh[layer].resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        velocity_bias_ih[layer].resize(4 * hidden_size, 0.0f);
        velocity_bias_hh[layer].resize(4 * hidden_size, 0.0f);
    }

    velocity_fc_weight.resize(num_classes, std::vector<float>(hidden_size, 0.0f));
    velocity_fc_bias.resize(num_classes, 0.0f);

    // Pre-allocate layer cache with minimal structure
    layer_cache.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        layer_cache[layer].resize(1);               
        layer_cache[layer][0].resize(lookback_len); 

        // Pre-allocate each cache entry with properly sized vectors
        for (auto &seq : layer_cache[layer][0]) {
            int expected_input_size = (layer == 0) ? input_size : hidden_size;
            seq.input.resize(expected_input_size, 0.0f);
            seq.prev_hidden.resize(hidden_size, 0.0f);
            seq.prev_cell.resize(hidden_size, 0.0f);
            seq.cell_state.resize(hidden_size, 0.0f);
            seq.input_gate.resize(hidden_size, 0.0f);
            seq.forget_gate.resize(hidden_size, 0.0f);
            seq.cell_candidate.resize(hidden_size, 0.0f);
            seq.output_gate.resize(hidden_size, 0.0f);
            seq.hidden_state.resize(hidden_size, 0.0f);
        }
    }

    // Pre-allocate gradients
    for (auto &grad : last_gradients) {
        int input_size_layer = (current_layer == 0) ? input_size : hidden_size;
        grad.weight_ih_grad.resize(4 * hidden_size, std::vector<float>(input_size_layer, 0.0f));
        grad.weight_hh_grad.resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        grad.bias_ih_grad.resize(4 * hidden_size, 0.0f);
        grad.bias_hh_grad.resize(4 * hidden_size, 0.0f);
    }

    // Pre-allocate states
    h_state.resize(num_layers, std::vector<float>(hidden_size, 0.0f));
    c_state.resize(num_layers, std::vector<float>(hidden_size, 0.0f));

    // Initialize weights
    initialize_weights();
    reset_states();

    std::cout << "LSTM Predictor initialization complete" << std::endl;
}

std::vector<float> LSTMPredictorNEON::forward_lstm_cell(const std::vector<float> &input,
                                                        std::vector<float> &h_state,
                                                        std::vector<float> &c_state,
                                                        const LSTMLayer &layer) {

    // Verify dimensions
    int expected_layer_input = (current_layer == 0) ? input_size : hidden_size;

    if (layer.weight_ih[0].size() != expected_layer_input) {
        throw std::runtime_error("Weight dimension mismatch in lstm_cell_forward_neon");
    }
    if (input.size() != expected_layer_input) {
        throw std::runtime_error("Input size mismatch in lstm_cell_forward_neon");
    }

    // Resize state vectors if needed
    if (h_state.size() != hidden_size)
        h_state.resize(hidden_size, 0.0f);
    if (c_state.size() != hidden_size)
        c_state.resize(hidden_size, 0.0f);

    // Cache entry for training mode
    LSTMCacheEntry *cache_entry = nullptr;
    if (training_mode) {
        // Validate cache access
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
    std::vector<float> gates(4 * hidden_size);

    // Add biases
    for (int h = 0; h + 3 < hidden_size; h += 4) {
        // Load bias vectors
        float32x4_t v_bias_ih_i = vld1q_f32(&layer.bias_ih[h]);
        float32x4_t v_bias_hh_i = vld1q_f32(&layer.bias_hh[h]);
        float32x4_t v_bias_ih_f = vld1q_f32(&layer.bias_ih[hidden_size + h]);
        float32x4_t v_bias_hh_f = vld1q_f32(&layer.bias_hh[hidden_size + h]);
        float32x4_t v_bias_ih_g = vld1q_f32(&layer.bias_ih[2 * hidden_size + h]);
        float32x4_t v_bias_hh_g = vld1q_f32(&layer.bias_hh[2 * hidden_size + h]);
        float32x4_t v_bias_ih_o = vld1q_f32(&layer.bias_ih[3 * hidden_size + h]);
        float32x4_t v_bias_hh_o = vld1q_f32(&layer.bias_hh[3 * hidden_size + h]);

        // Add corresponding biases
        float32x4_t v_gate_i = vaddq_f32(v_bias_ih_i, v_bias_hh_i);
        float32x4_t v_gate_f = vaddq_f32(v_bias_ih_f, v_bias_hh_f);
        float32x4_t v_gate_g = vaddq_f32(v_bias_ih_g, v_bias_hh_g);
        float32x4_t v_gate_o = vaddq_f32(v_bias_ih_o, v_bias_hh_o);

        // Store in gates vector
        vst1q_f32(&gates[h], v_gate_i);
        vst1q_f32(&gates[hidden_size + h], v_gate_f);
        vst1q_f32(&gates[2 * hidden_size + h], v_gate_g);
        vst1q_f32(&gates[3 * hidden_size + h], v_gate_o);
    }

    // Handle remaining elements
    for (int h = (hidden_size / 4) * 4; h < hidden_size; ++h) {
        gates[h] = layer.bias_ih[h] + layer.bias_hh[h]; // input gate
        gates[hidden_size + h] =
            layer.bias_ih[hidden_size + h] + layer.bias_hh[hidden_size + h]; // forget gate
        gates[2 * hidden_size + h] = layer.bias_ih[2 * hidden_size + h] +
                                     layer.bias_hh[2 * hidden_size + h]; // cell candidate
        gates[3 * hidden_size + h] =
            layer.bias_ih[3 * hidden_size + h] + layer.bias_hh[3 * hidden_size + h]; // output gate
    }

    // Input to hidden contribution
    for (size_t i = 0; i < input.size(); ++i) {
        // Broadcast input element
        float32x4_t v_input = vdupq_n_f32(input[i]);

        // Process each gate type
        for (int gate_type = 0; gate_type < 4; ++gate_type) {
            int gate_offset = gate_type * hidden_size;

            // Process 4 hidden units at a time
            for (int h = 0; h + 3 < hidden_size; h += 4) {
                // Load weights and gates
                float32x4_t v_weights = vld1q_f32(&layer.weight_ih[gate_offset + h][i]);
                float32x4_t v_gates = vld1q_f32(&gates[gate_offset + h]);

                // Multiply-accumulate: gates += weights * input
                v_gates = vmlaq_f32(v_gates, v_weights, v_input);

                // Store updated gates
                vst1q_f32(&gates[gate_offset + h], v_gates);
            }

            // Handle remaining units
            for (int h = (hidden_size / 4) * 4; h < hidden_size; ++h) {
                gates[gate_offset + h] += layer.weight_ih[gate_offset + h][i] * input[i];
            }
        }
    }

    // Hidden to hidden contribution
    for (size_t i = 0; i < hidden_size; ++i) {
        // Broadcast h_state element
        float32x4_t v_h_state = vdupq_n_f32(h_state[i]);

        // Process each gate type
        for (int gate_type = 0; gate_type < 4; ++gate_type) {
            int gate_offset = gate_type * hidden_size;

            // Process 4 hidden units at a time
            for (int h = 0; h + 3 < hidden_size; h += 4) {
                // Load weights and gates
                float32x4_t v_weights = vld1q_f32(&layer.weight_hh[gate_offset + h][i]);
                float32x4_t v_gates = vld1q_f32(&gates[gate_offset + h]);

                // Multiply-accumulate: gates += weights * h_state
                v_gates = vmlaq_f32(v_gates, v_weights, v_h_state);

                // Store updated gates
                vst1q_f32(&gates[gate_offset + h], v_gates);
            }

            // Handle remaining units
            for (int h = (hidden_size / 4) * 4; h < hidden_size; ++h) {
                gates[gate_offset + h] += layer.weight_hh[gate_offset + h][i] * h_state[i];
            }
        }
    }

    // Apply activations and update states
    std::vector<float> output(hidden_size);

    // Process 4 elements at a time
    for (int h = 0; h + 3 < hidden_size; h += 4) {
        // Load gate values
        float32x4_t v_gate_i = vld1q_f32(&gates[h]);
        float32x4_t v_gate_f = vld1q_f32(&gates[hidden_size + h]);
        float32x4_t v_gate_g = vld1q_f32(&gates[2 * hidden_size + h]);
        float32x4_t v_gate_o = vld1q_f32(&gates[3 * hidden_size + h]);

        // Apply activations
        float32x4_t v_i_t = sigmoid_neon(v_gate_i);
        float32x4_t v_f_t = sigmoid_neon(v_gate_f);
        float32x4_t v_cell_candidate = tanh_neon(v_gate_g);
        float32x4_t v_o_t = sigmoid_neon(v_gate_o);

        // Load cell state
        float32x4_t v_c_state = vld1q_f32(&c_state[h]);

        // Compute new cell state: c_t = f_t * c_{t-1} + i_t * g_t
        float32x4_t v_fc = vmulq_f32(v_f_t, v_c_state);
        float32x4_t v_ig = vmulq_f32(v_i_t, v_cell_candidate);
        float32x4_t v_new_c = vaddq_f32(v_fc, v_ig);

        // Compute tanh of new cell state
        float32x4_t v_tanh_c = tanh_neon(v_new_c);

        // Compute new hidden state: h_t = o_t * tanh(c_t)
        float32x4_t v_new_h = vmulq_f32(v_o_t, v_tanh_c);

        // Store new states
        vst1q_f32(&c_state[h], v_new_c);
        vst1q_f32(&h_state[h], v_new_h);
        vst1q_f32(&output[h], v_new_h);

        // Store in cache if in training mode
        if (training_mode && cache_entry) {
            vst1q_f32(&cache_entry->input_gate[h], v_i_t);
            vst1q_f32(&cache_entry->forget_gate[h], v_f_t);
            vst1q_f32(&cache_entry->cell_candidate[h], v_cell_candidate);
            vst1q_f32(&cache_entry->output_gate[h], v_o_t);
            vst1q_f32(&cache_entry->cell_state[h], v_new_c);
            vst1q_f32(&cache_entry->hidden_state[h], v_new_h);
        }
    }

    // Handle remaining elements
    for (int h = (hidden_size / 4) * 4; h < hidden_size; ++h) {
        float i_t = sigmoid(gates[h]);
        float f_t = sigmoid(gates[hidden_size + h]);
        float cell_candidate = tanh_custom(gates[2 * hidden_size + h]);
        float o_t = sigmoid(gates[3 * hidden_size + h]);

        // Update cell state
        float new_cell = f_t * c_state[h] + i_t * cell_candidate;
        c_state[h] = new_cell;

        // Update hidden state
        float new_hidden = o_t * tanh_custom(new_cell);
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

LSTMPredictor::LSTMOutput LSTMPredictorNEON::forward_lstm(const std::vector<std::vector<std::vector<float>>> &x,
                           const std::vector<std::vector<float>> *initial_hidden,
                           const std::vector<std::vector<float>> *initial_cell) {

    // Verify input dimensions (same as base class)
    for (size_t batch = 0; batch < x.size(); ++batch) {
        for (size_t seq = 0; seq < x[batch].size(); ++seq) {
            if (x[batch][seq].size() != input_size) {
                throw std::runtime_error("Input dimension mismatch in sequence");
            }
        }
    }

    try {
        // Initialize layer cache if not already initialized
        if (!is_layer_cache_initialized()) {
            initialize_layer_cache();
        }

        // Verify cache is properly initialized
        if (layer_cache.size() != num_layers) {
            throw std::runtime_error("Layer cache not properly initialized");
        }

        size_t batch_size = x.size();
        size_t seq_len = x[0].size();

        LSTMOutput output;
        output.sequence_output.resize(batch_size);

        for (auto &batch : output.sequence_output) {
            batch.resize(seq_len);
            for (auto &seq : batch) {
                seq.resize(hidden_size);

                // Initializing with zeros in blocks of 4
                size_t i = 0;
                float32x4_t v_zero = vdupq_n_f32(0.0f);

                for (; i + 3 < hidden_size; i += 4) {
                    vst1q_f32(&seq[i], v_zero);
                }

                // Handle remaining elements
                for (; i < hidden_size; ++i) {
                    seq[i] = 0.0f;
                }
            }
        }

        // Process each batch
        for (size_t batch = 0; batch < batch_size; ++batch) {
            current_batch = batch;

            // Reset states if no initial states provided
            if (!initial_hidden || !initial_cell) {
                reset_states();
            }

            // Process each time step
            for (size_t t = 0; t < seq_len; ++t) {
                current_timestep = t;

                std::vector<float> layer_input = x[batch][t];
                std::vector<std::vector<float>> layer_outputs(num_layers + 1);

                // Copy vector
                layer_outputs[0].resize(layer_input.size());

                size_t i = 0;
                for (; i + 3 < layer_input.size(); i += 4) {
                    float32x4_t v_input = vld1q_f32(&layer_input[i]);
                    vst1q_f32(&layer_outputs[0][i], v_input);
                }

                // Handle remaining elements
                for (; i < layer_input.size(); ++i) {
                    layer_outputs[0][i] = layer_input[i];
                }

                // Process through LSTM layers
                for (int layer = 0; layer < num_layers; ++layer) {
                    current_layer = layer;

                    // Get correct input size for this layer
                    int expected_input_size = (layer == 0) ? input_size : hidden_size;

                    // Verify dimensions
                    if (layer_outputs[layer].size() != expected_input_size) {
                        throw std::runtime_error("Layer input dimension mismatch at layer " +
                                                 std::to_string(static_cast<long long>(layer)));
                    }

                    // Ensure cache entry exists and is properly sized if in training mode
                    if (training_mode) {
                        // Validate indices before accessing cache
                        if (current_layer >= layer_cache.size() ||
                            current_batch >= layer_cache[current_layer].size() ||
                            current_timestep >= layer_cache[current_layer][current_batch].size()) {
                            throw std::runtime_error("Invalid cache access");
                        }

                        auto &cache_entry =
                            layer_cache[current_layer][current_batch][current_timestep];

                        // Resize and initialize to zero
                        cache_entry.input.resize(expected_input_size);
                        cache_entry.prev_hidden.resize(hidden_size);
                        cache_entry.prev_cell.resize(hidden_size);
                        cache_entry.cell_state.resize(hidden_size);
                        cache_entry.input_gate.resize(hidden_size);
                        cache_entry.forget_gate.resize(hidden_size);
                        cache_entry.cell_candidate.resize(hidden_size);
                        cache_entry.output_gate.resize(hidden_size);
                        cache_entry.hidden_state.resize(hidden_size);
                    }

                    layer_outputs[layer + 1] = forward_lstm_cell(
                        layer_outputs[layer], h_state[layer], c_state[layer], lstm_layers[layer]);
                }

                // Copy final layer output to sequence output
                size_t j = 0;
                for (; j + 3 < hidden_size; j += 4) {
                    float32x4_t v_output = vld1q_f32(&layer_outputs[num_layers][j]);
                    vst1q_f32(&output.sequence_output[batch][t][j], v_output);
                }

                // Handle remaining elements
                for (; j < hidden_size; ++j) {
                    output.sequence_output[batch][t][j] = layer_outputs[num_layers][j];
                }
            }
        }

        output.final_hidden = h_state;
        output.final_cell = c_state;

        return output;

    } catch (const std::exception &e) {
        // Clean up on error
        clear_update_state();
        throw;
    }
}

void LSTMPredictorNEON::backward_linear_layer(const std::vector<float> &grad_output,
                                              const std::vector<float> &last_hidden,
                                              std::vector<std::vector<float>> &weight_grad,
                                              std::vector<float> &bias_grad,
                                              std::vector<float> &input_grad) {

    // Check dimensions (same as base class)
    if (grad_output.size() != num_classes) {
        throw std::invalid_argument(
            "grad_output size mismatch: " + std::to_string(grad_output.size()) +
            " != " + std::to_string(num_classes));
    }

    if (last_hidden.size() != hidden_size) {
        throw std::invalid_argument(
            "last_hidden size mismatch: " + std::to_string(last_hidden.size()) +
            " != " + std::to_string(hidden_size));
    }

    // Initialize gradients with correct dimensions
    weight_grad.resize(num_classes, std::vector<float>(hidden_size, 0.0f));
    bias_grad = grad_output; // Copy gradient directly for bias
    input_grad.resize(hidden_size, 0.0f);

    // Compute weight gradients
    for (int i = 0; i < num_classes; ++i) {
        // Broadcast grad_output[i] to a vector of 4 identical values
        float32x4_t v_grad_output_i = vdupq_n_f32(grad_output[i]);

        int j = 0;
        // Process 4 hidden units at a time
        for (; j + 3 < hidden_size; j += 4) {
            // Load 4 values from last_hidden
            float32x4_t v_last_hidden = vld1q_f32(&last_hidden[j]);

            // Multiply gradient with last hidden (element-wise)
            float32x4_t v_result = vmulq_f32(v_grad_output_i, v_last_hidden);

            // Store the result to weight_grad
            vst1q_f32(&weight_grad[i][j], v_result);
        }

        // Handle remaining elements
        for (; j < hidden_size; ++j) {
            weight_grad[i][j] = grad_output[i] * last_hidden[j];
        }
    }

    // Compute input gradients
    // input_grad = fc_weight^T * grad_output
    float32x4_t v_zeros = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < hidden_size; i += 4) {
        vst1q_f32(&input_grad[i], v_zeros);
    }
    for (; i < hidden_size; ++i) {
        input_grad[i] = 0.0f;
    }

    for (int j = 0; j < num_classes; ++j) {
        // Broadcast grad_output[j]
        float32x4_t v_grad_j = vdupq_n_f32(grad_output[j]);

        i = 0;
        for (; i + 3 < hidden_size; i += 4) {
            float32x4_t v_input_grad = vld1q_f32(&input_grad[i]);
            float32x4_t v_weights = vld1q_f32(&fc_weight[j][i]);
            v_input_grad = vmlaq_f32(v_input_grad, v_weights, v_grad_j);
            vst1q_f32(&input_grad[i], v_input_grad);
        }

        // Handle remaining elements
        for (; i < hidden_size; ++i) {
            input_grad[i] += fc_weight[j][i] * grad_output[j];
        }
    }
}

std::vector<LSTMPredictor::LSTMGradients> LSTMPredictorNEON::backward_lstm_layer(
    const std::vector<float> &grad_output,
    const std::vector<std::vector<std::vector<LSTMCacheEntry>>> &cache, float learning_rate) {

    // Dimension validation
    if (grad_output.size() != hidden_size) {
        throw std::runtime_error("grad_output size mismatch in backward_lstm_layer_neon");
    }
    if (cache.size() != num_layers) {
        throw std::runtime_error("cache layer count mismatch in backward_lstm_layer_neon");
    }

    // Initialize gradients for each layer
    std::vector<LSTMGradients> layer_grads(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        layer_grads[layer].weight_ih_grad.resize(4 * hidden_size,
                                                 std::vector<float>(input_size_layer, 0.0f));
        layer_grads[layer].weight_hh_grad.resize(4 * hidden_size,
                                                 std::vector<float>(hidden_size, 0.0f));
        layer_grads[layer].bias_ih_grad.resize(4 * hidden_size, 0.0f);
        layer_grads[layer].bias_hh_grad.resize(4 * hidden_size, 0.0f);
    }

    // Initialize dh_next and dc_next for each layer
    std::vector<std::vector<float>> dh_next(num_layers, std::vector<float>(hidden_size, 0.0f));
    std::vector<std::vector<float>> dc_next(num_layers, std::vector<float>(hidden_size, 0.0f));

    // Start from last layer and move backward
    for (int layer = num_layers - 1; layer >= 0; --layer) {
        // Get dh and dc for this layer
        std::vector<float> dh = dh_next[layer];
        std::vector<float> dc = dc_next[layer];

        // Validate cache access
        if (current_batch >= cache[layer].size()) {
            throw std::runtime_error("Cache batch index out of bounds");
        }

        const auto &layer_cache = cache[layer][current_batch];

        // If this is the last layer, add grad_output to dh
        if (layer == num_layers - 1) {
            // Use NEON to add grad_output to dh
            size_t h = 0;
            for (; h + 3 < hidden_size; h += 4) {
                float32x4_t v_dh = vld1q_f32(&dh[h]);
                float32x4_t v_grad = vld1q_f32(&grad_output[h]);
                v_dh = vaddq_f32(v_dh, v_grad);
                vst1q_f32(&dh[h], v_dh);
            }

            // Handle remaining elements
            for (; h < hidden_size; ++h) {
                dh[h] += grad_output[h];
            }
        }

        // Process each time step in reverse order
        for (int t = layer_cache.size() - 1; t >= 0; --t) {
            const auto &cache_entry = layer_cache[t];

            // Initialize dh_prev and dc_prev
            std::vector<float> dh_prev(hidden_size, 0.0f);
            std::vector<float> dc_prev(hidden_size, 0.0f);

            // Calculate gradients for this time step
            size_t h = 0;

            // Process 4 hidden units at a time
            for (; h + 3 < hidden_size; h += 4) {
                // Constants
                float32x4_t v_one = vdupq_n_f32(1.0f);

                // Load cell state and compute tanh(c)
                float32x4_t v_cell_state = vld1q_f32(&cache_entry.cell_state[h]);
                float32x4_t v_tanh_c = tanh_neon(v_cell_state);

                // Load dh
                float32x4_t v_dh = vld1q_f32(&dh[h]);

                // Load gates
                float32x4_t v_output_gate = vld1q_f32(&cache_entry.output_gate[h]);
                float32x4_t v_input_gate = vld1q_f32(&cache_entry.input_gate[h]);
                float32x4_t v_forget_gate = vld1q_f32(&cache_entry.forget_gate[h]);
                float32x4_t v_cell_candidate = vld1q_f32(&cache_entry.cell_candidate[h]);
                float32x4_t v_prev_cell = vld1q_f32(&cache_entry.prev_cell[h]);

                // Load dc
                float32x4_t v_dc = vld1q_f32(&dc[h]);

                // Compute tanh'(c_t) = 1 - tanh(c_t)²
                float32x4_t v_tanh_c_squared = vmulq_f32(v_tanh_c, v_tanh_c);
                float32x4_t v_dtanh_c = vsubq_f32(v_one, v_tanh_c_squared);

                // Compute dct = dht * ot * tanh'(ct) + dct+1
                float32x4_t v_dho_dtanh = vmulq_f32(v_dh, v_output_gate);
                v_dho_dtanh = vmulq_f32(v_dho_dtanh, v_dtanh_c);
                float32x4_t v_dc_t = vaddq_f32(v_dho_dtanh, v_dc);

                // Compute gradients for gates
                // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

                // Output gate gradient: dot = dht * tanh(ct) * ot * (1 - ot)
                float32x4_t v_do_t = vmulq_f32(v_dh, v_tanh_c);
                float32x4_t v_one_minus_o = vsubq_f32(v_one, v_output_gate);
                v_do_t = vmulq_f32(v_do_t, vmulq_f32(v_output_gate, v_one_minus_o));

                // Input gate gradient: dit = dct * gt * it * (1 - it)
                float32x4_t v_di_t = vmulq_f32(v_dc_t, v_cell_candidate);
                float32x4_t v_one_minus_i = vsubq_f32(v_one, v_input_gate);
                v_di_t = vmulq_f32(v_di_t, vmulq_f32(v_input_gate, v_one_minus_i));

                // Forget gate gradient: dft = dct * ct-1 * ft * (1 - ft)
                float32x4_t v_df_t = vmulq_f32(v_dc_t, v_prev_cell);
                float32x4_t v_one_minus_f = vsubq_f32(v_one, v_forget_gate);
                v_df_t = vmulq_f32(v_df_t, vmulq_f32(v_forget_gate, v_one_minus_f));

                // Cell candidate gradient: dgt = dct * it * (1 - gt²)
                float32x4_t v_g_squared = vmulq_f32(v_cell_candidate, v_cell_candidate);
                float32x4_t v_one_minus_g_squared = vsubq_f32(v_one, v_g_squared);
                float32x4_t v_dg_t = vmulq_f32(v_dc_t, vmulq_f32(v_input_gate, v_one_minus_g_squared));

                // Store gate gradients temporarily to use later
                float di_t[4], df_t[4], dg_t[4], do_t[4], dc_t[4];
                vst1q_f32(di_t, v_di_t);
                vst1q_f32(df_t, v_df_t);
                vst1q_f32(dg_t, v_dg_t);
                vst1q_f32(do_t, v_do_t);
                vst1q_f32(dc_t, v_dc_t);

                // Accumulate bias gradients
                for (int idx = 0; idx < 4; ++idx) {
                    layer_grads[layer].bias_ih_grad[h + idx] += di_t[idx];
                    layer_grads[layer].bias_ih_grad[hidden_size + h + idx] += df_t[idx];
                    layer_grads[layer].bias_ih_grad[2 * hidden_size + h + idx] += dg_t[idx];
                    layer_grads[layer].bias_ih_grad[3 * hidden_size + h + idx] += do_t[idx];

                    // Compute dc_prev
                    dc_prev[h + idx] = dc_t[idx] * cache_entry.forget_gate[h + idx];
                }
            }

            // Handle remaining elements
            for (; h < hidden_size; ++h) {
                float tanh_c = tanh_custom(cache_entry.cell_state[h]);
                float dho = dh[h];

                // Cell state gradient
                float dc_t = dho * cache_entry.output_gate[h] * (1.0f - tanh_c * tanh_c);
                dc_t += dc[h];

                // Gate gradients
                float do_t =
                    dho * tanh_c * cache_entry.output_gate[h] * (1.0f - cache_entry.output_gate[h]);
                float di_t = dc_t * cache_entry.cell_candidate[h] * cache_entry.input_gate[h] *
                             (1.0f - cache_entry.input_gate[h]);
                float df_t = dc_t * cache_entry.prev_cell[h] * cache_entry.forget_gate[h] *
                             (1.0f - cache_entry.forget_gate[h]);
                float dg_t = dc_t * cache_entry.input_gate[h] *
                             (1.0f - cache_entry.cell_candidate[h] * cache_entry.cell_candidate[h]);

                // Accumulate bias gradients
                layer_grads[layer].bias_ih_grad[h] += di_t;
                layer_grads[layer].bias_ih_grad[hidden_size + h] += df_t;
                layer_grads[layer].bias_ih_grad[2 * hidden_size + h] += dg_t;
                layer_grads[layer].bias_ih_grad[3 * hidden_size + h] += do_t;

                // Cell state gradient for previous timestep
                dc_prev[h] = dc_t * cache_entry.forget_gate[h];
            }

            // Accumulate weight gradients and compute dh_prev
            // Difficult to vectorize this part due to the nested loops and non-contiguous memory
            int input_size_layer = (layer == 0) ? input_size : hidden_size;

            for (int h = 0; h < hidden_size; ++h) {
                float tanh_c = tanh_custom(cache_entry.cell_state[h]);
                float dho = dh[h];

                // Recompute gradients (more compact than storing all of them)
                float dc_t = dho * cache_entry.output_gate[h] * (1.0f - tanh_c * tanh_c) + dc[h];
                float do_t = dho * tanh_c * cache_entry.output_gate[h] * (1.0f - cache_entry.output_gate[h]);
                float di_t = dc_t * cache_entry.cell_candidate[h] * cache_entry.input_gate[h] * (1.0f - cache_entry.input_gate[h]);
                float df_t = dc_t * cache_entry.prev_cell[h] * cache_entry.forget_gate[h] * (1.0f - cache_entry.forget_gate[h]);
                float dg_t = dc_t * cache_entry.input_gate[h] * (1.0f - cache_entry.cell_candidate[h] * cache_entry.cell_candidate[h]);

                // Accumulate weight gradients
                for (int j = 0; j < input_size_layer; ++j) {
                    float input_j = cache_entry.input[j];
                    layer_grads[layer].weight_ih_grad[h][j] += di_t * input_j;
                    layer_grads[layer].weight_ih_grad[hidden_size + h][j] += df_t * input_j;
                    layer_grads[layer].weight_ih_grad[2 * hidden_size + h][j] += dg_t * input_j;
                    layer_grads[layer].weight_ih_grad[3 * hidden_size + h][j] += do_t * input_j;
                }

                // Accumulate hidden-hidden weight gradients and dh_prev
                for (int j = 0; j < hidden_size; ++j) {
                    float h_prev_j = cache_entry.prev_hidden[j];
                    layer_grads[layer].weight_hh_grad[h][j] += di_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[hidden_size + h][j] += df_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[2 * hidden_size + h][j] += dg_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[3 * hidden_size + h][j] += do_t * h_prev_j;

                    // Accumulate dh_prev
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

// SGD update for weight matrices
void LSTMPredictorNEON::apply_sgd_update(std::vector<std::vector<float>> &weights,
                                         std::vector<std::vector<float>> &grads,
                                         float learning_rate, float momentum) {

    // Find which velocity terms to use based on dimensions
    std::vector<std::vector<float>> *velocity_terms = nullptr;
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
        throw std::runtime_error("Unknown weight dimensions in apply_sgd_update_neon");
    }

    // NEON constants for vectorized operations
    float32x4_t v_momentum = vdupq_n_f32(momentum);     // Load momentum as vector
    float32x4_t v_neg_lr = vdupq_n_f32(-learning_rate); // Negative learning rate
    float32x4_t v_clip_max = vdupq_n_f32(1.0f);         // Clip max value
    float32x4_t v_clip_min = vdupq_n_f32(-1.0f);        // Clip min value

    // Process each row of weights
    for (size_t i = 0; i < weights.size(); ++i) {
        size_t j = 0;

        // Process 4 elements at once
        for (; j + 3 < weights[i].size(); j += 4) {
            // Load gradients and velocity
            float32x4_t v_grad = vld1q_f32(&grads[i][j]);
            float32x4_t v_velocity = vld1q_f32(&(*velocity_terms)[i][j]);

            // Clip gradients between -1.0 and 1.0
            v_grad = vminq_f32(v_grad, v_clip_max);
            v_grad = vmaxq_f32(v_grad, v_clip_min);

            // Update velocity: v = momentum * v - lr * grad
            float32x4_t v_scaled_grad = vmulq_f32(v_grad, v_neg_lr);
            float32x4_t v_momentum_vel = vmulq_f32(v_velocity, v_momentum);
            v_velocity = vaddq_f32(v_momentum_vel, v_scaled_grad);

            // Store updated velocity
            vst1q_f32(&(*velocity_terms)[i][j], v_velocity);

            // Load weights
            float32x4_t v_weight = vld1q_f32(&weights[i][j]);

            // Update weights: w = w + v
            v_weight = vaddq_f32(v_weight, v_velocity);

            // Store updated weights
            vst1q_f32(&weights[i][j], v_weight);
        }

        // Handle remaining elements (less than 4)
        for (; j < weights[i].size(); ++j) {
            float grad = grads[i][j];

            // Gradient clipping
            grad = std::max(std::min(grad, 1.0f), -1.0f);

            // Update velocity with momentum
            float velocity = momentum * (*velocity_terms)[i][j] - learning_rate * grad;
            (*velocity_terms)[i][j] = velocity;

            // Update weights using velocity
            weights[i][j] += velocity;
        }
    }
}

// SGD update for bias vectors
void LSTMPredictorNEON::apply_sgd_update(std::vector<float> &biases, std::vector<float> &grads,
                                         float learning_rate, float momentum) {

    // Find which velocity terms to use
    std::vector<float> *velocity_terms = nullptr;

    if (biases.size() == num_classes) {
        velocity_terms = &velocity_fc_bias;
    } else if (biases.size() == 4 * hidden_size) {
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
        throw std::runtime_error("Unknown bias dimensions in apply_sgd_update_neon");
    }

    // NEON constants
    float32x4_t v_momentum = vdupq_n_f32(momentum);
    float32x4_t v_neg_lr = vdupq_n_f32(-learning_rate);
    float32x4_t v_clip_max = vdupq_n_f32(1.0f);
    float32x4_t v_clip_min = vdupq_n_f32(-1.0f);

    size_t i = 0;

    // Process 4 elements at a time
    for (; i + 3 < biases.size(); i += 4) {
        // Load gradients and velocity
        float32x4_t v_grad = vld1q_f32(&grads[i]);
        float32x4_t v_velocity = vld1q_f32(&(*velocity_terms)[i]);

        // Clip gradients
        v_grad = vminq_f32(v_grad, v_clip_max);
        v_grad = vmaxq_f32(v_grad, v_clip_min);

        // Compute new velocity
        float32x4_t v_scaled_grad = vmulq_f32(v_grad, v_neg_lr);
        float32x4_t v_momentum_vel = vmulq_f32(v_velocity, v_momentum);
        v_velocity = vaddq_f32(v_momentum_vel, v_scaled_grad);

        // Store updated velocity
        vst1q_f32(&(*velocity_terms)[i], v_velocity);

        // Load biases
        float32x4_t v_bias = vld1q_f32(&biases[i]);

        // Update biases
        v_bias = vaddq_f32(v_bias, v_velocity);

        // Store updated biases
        vst1q_f32(&biases[i], v_bias);
    }

    // Handle remaining elements
    for (; i < biases.size(); ++i) {
        float grad = grads[i];

        // Gradient clipping
        grad = std::max(std::min(grad, 1.0f), -1.0f);

        // Update velocity with momentum
        float velocity = momentum * (*velocity_terms)[i] - learning_rate * grad;
        (*velocity_terms)[i] = velocity;

        // Update biases using velocity
        biases[i] += velocity;
    }
}

std::vector<float> LSTMPredictorNEON::forward_linear(const LSTMOutput &lstm_output) {
    std::vector<float> final_output(num_classes, 0.0f);

    // Get final hidden state
    const auto &final_hidden = lstm_output.sequence_output.back().back();

    // For each output class
    for (int i = 0; i < num_classes; ++i) {
        float sum = fc_bias[i];
        int j = 0;
        float32x4_t v_sum = vdupq_n_f32(0.0f);

        // Process 4 hidden units at a time
        for (; j + 3 < hidden_size; j += 4) {
            // Load weights and hidden state
            float32x4_t v_weight = vld1q_f32(&fc_weight[i][j]);
            float32x4_t v_hidden = vld1q_f32(&final_hidden[j]);

            // Multiply-accumulate
            v_sum = vmlaq_f32(v_sum, v_weight, v_hidden);
        }

        // Reduce vector sum to scalar
        float32x2_t v_sum_low = vget_low_f32(v_sum);
        float32x2_t v_sum_high = vget_high_f32(v_sum);
        v_sum_low = vadd_f32(v_sum_low, v_sum_high);
        sum += vget_lane_f32(vpadd_f32(v_sum_low, v_sum_low), 0);

        // Handle remaining elements
        for (; j < hidden_size; ++j) {
            sum += fc_weight[i][j] * final_hidden[j];
        }

        final_output[i] = sum;
    }

    return final_output;
}

void LSTMPredictorNEON::apply_gate_operations_neon(std::vector<float> &gates,
                                                   std::vector<float> &h_state,
                                                   std::vector<float> &c_state,
                                                   LSTMCacheEntry *cache_entry,
                                                   size_t hidden_size) {

    // Process 4 elements at a time
    for (size_t h = 0; h + 3 < hidden_size; h += 4) {
        // Load gate values
        float32x4_t v_i_gate = vld1q_f32(&gates[h]);
        float32x4_t v_f_gate = vld1q_f32(&gates[hidden_size + h]);
        float32x4_t v_g_gate = vld1q_f32(&gates[2 * hidden_size + h]);
        float32x4_t v_o_gate = vld1q_f32(&gates[3 * hidden_size + h]);

        // Apply activations
        float32x4_t v_i_t = sigmoid_neon(v_i_gate);
        float32x4_t v_f_t = sigmoid_neon(v_f_gate);
        float32x4_t v_g_t = tanh_neon(v_g_gate);
        float32x4_t v_o_t = sigmoid_neon(v_o_gate);

        // Load cell state
        float32x4_t v_c_state = vld1q_f32(&c_state[h]);

        // Compute new cell state: c_t = f_t * c_{t-1} + i_t * g_t
        float32x4_t v_fc = vmulq_f32(v_f_t, v_c_state);
        float32x4_t v_ig = vmulq_f32(v_i_t, v_g_t);
        float32x4_t v_new_c = vaddq_f32(v_fc, v_ig);

        // Compute tanh of new cell state
        float32x4_t v_tanh_c = tanh_neon(v_new_c);

        // Compute new hidden state: h_t = o_t * tanh(c_t)
        float32x4_t v_new_h = vmulq_f32(v_o_t, v_tanh_c);

        // Store new states
        vst1q_f32(&c_state[h], v_new_c);
        vst1q_f32(&h_state[h], v_new_h);

        // Store in cache if in training mode
        if (cache_entry) {
            vst1q_f32(&cache_entry->input_gate[h], v_i_t);
            vst1q_f32(&cache_entry->forget_gate[h], v_f_t);
            vst1q_f32(&cache_entry->cell_candidate[h], v_g_t);
            vst1q_f32(&cache_entry->output_gate[h], v_o_t);
            vst1q_f32(&cache_entry->cell_state[h], v_new_c);
            vst1q_f32(&cache_entry->hidden_state[h], v_new_h);
        }
    }

    // Handle remaining elements
    for (size_t h = (hidden_size / 4) * 4; h < hidden_size; ++h) {
        // Apply scalar operations for remaining elements
        float i_t = this->sigmoid(gates[h]);
        float f_t = this->sigmoid(gates[hidden_size + h]);
        float g_t = this->tanh_custom(gates[2 * hidden_size + h]);
        float o_t = this->sigmoid(gates[3 * hidden_size + h]);

        // Update cell state
        float new_c = f_t * c_state[h] + i_t * g_t;
        c_state[h] = new_c;

        // Update hidden state
        float new_h = o_t * this->tanh_custom(new_c);
        h_state[h] = new_h;

        // Store in cache if in training mode
        if (cache_entry) {
            cache_entry->input_gate[h] = i_t;
            cache_entry->forget_gate[h] = f_t;
            cache_entry->cell_candidate[h] = g_t;
            cache_entry->output_gate[h] = o_t;
            cache_entry->cell_state[h] = new_c;
            cache_entry->hidden_state[h] = new_h;
        }
    }
}

void LSTMPredictorNEON::mse_loss_gradient(const std::vector<float> &output,
                                    const std::vector<float> &target,
                                    std::vector<float> &gradient) {
    // Ensure gradient is properly sized
    if (gradient.size() != output.size()) {
        gradient.resize(output.size());
    }
    
    const size_t size = output.size();
    const float scale = 2.0f / size;
    
    // Process 4 elements at a time using NEON
    size_t i = 0;
    const size_t vec_size = size - (size % 4);
    
    // Preload scale factor into a NEON register
    float32x4_t v_scale = vdupq_n_f32(scale);
    
    for (; i < vec_size; i += 4) {
        // Load 4 elements from output and target
        float32x4_t v_output = vld1q_f32(&output[i]);
        float32x4_t v_target = vld1q_f32(&target[i]);
        
        // Calculate difference: (output - target)
        float32x4_t v_diff = vsubq_f32(v_output, v_target);
        
        // Calculate gradient: 2.0f * (output - target) / size
        float32x4_t v_gradient = vmulq_f32(v_diff, v_scale);
        
        // Store the result
        vst1q_f32(&gradient[i], v_gradient);
    }
    
    // Handle remaining elements (if any)
    for (; i < size; ++i) {
        gradient[i] = scale * (output[i] - target[i]);
    }
}

float LSTMPredictorNEON::mse_loss(const std::vector<float> &prediction,
                                  const std::vector<float> &target) {
    if (prediction.size() != target.size()) {
        throw std::runtime_error("Prediction and target size mismatch in mse_loss");
    }

    float32x4_t v_sum = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Process 4 elements at a time
    for (; i + 3 < prediction.size(); i += 4) {
        // Load prediction and target vectors
        float32x4_t v_pred = vld1q_f32(&prediction[i]);
        float32x4_t v_target = vld1q_f32(&target[i]);

        // Calculate difference
        float32x4_t v_diff = vsubq_f32(v_pred, v_target);

        // Square the difference and accumulate
        v_sum = vmlaq_f32(v_sum, v_diff, v_diff);
    }

    // Sum up the vector to get final result
    float32x2_t v_sum_low = vget_low_f32(v_sum);
    float32x2_t v_sum_high = vget_high_f32(v_sum);
    v_sum_low = vadd_f32(v_sum_low, v_sum_high);
    float32x2_t v_final = vpadd_f32(v_sum_low, v_sum_low);
    float loss = vget_lane_f32(v_final, 0);

    // Handle remaining elements
    for (; i < prediction.size(); ++i) {
        float diff = prediction[i] - target[i];
        loss += diff * diff;
    }

    return loss;
}