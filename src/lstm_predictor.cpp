#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <arm_neon.h>
#include <cmath>
#include <cstring>

#include <random>
#include <algorithm>
#include <iostream>

// NEON-optimized pow function for vectors of 4 elements
inline float32x4_t pow_float_neon(float32x4_t base, float32x4_t exp) {
    // Store vectors in temporary arrays
    float base_array[4], exp_array[4];
    vst1q_f32(base_array, base);
    vst1q_f32(exp_array, exp);
    
    // Compute pow using scalar operations
    float result_array[4];
    for (int i = 0; i < 4; i++) {
        result_array[i] = std::pow(base_array[i], exp_array[i]);
    }
    
    return vld1q_f32(result_array);
}

// Scalar pow function for remaining elements
inline float pow_float(float base, float exp) {
    return std::pow(base, exp);
}

// Custom implementations for missing NEON intrinsics
inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) {
    float a_array[4], b_array[4];
    vst1q_f32(a_array, a);
    vst1q_f32(b_array, b);
    for (int i = 0; i < 4; i++) {
        a_array[i] = a_array[i] / b_array[i];
    }
    return vld1q_f32(a_array);
}

inline float32x4_t vexpq_f32(float32x4_t x) {
    float x_array[4];
    vst1q_f32(x_array, x);
    for (int i = 0; i < 4; i++) {
        x_array[i] = std::exp(x_array[i]);
    }
    return vld1q_f32(x_array);
}

inline float32x4_t vlogq_f32(float32x4_t x) {
    float x_array[4];
    vst1q_f32(x_array, x);
    for (int i = 0; i < 4; i++) {
        x_array[i] = std::log(x_array[i]);
    }
    return vld1q_f32(x_array);
}

inline float32x4_t vsqrtq_f32(float32x4_t x) {
    float x_array[4];
    vst1q_f32(x_array, x);
    for (int i = 0; i < 4; i++) {
        x_array[i] = std::sqrt(x_array[i]);
    }
    return vld1q_f32(x_array);
}

LSTMPredictor::LSTMPredictor(int num_classes, int input_size, int hidden_size, 
                            int num_layers, int lookback_len, 
                            bool batch_first)
    : num_classes(num_classes),
      num_layers(num_layers),
      input_size(input_size),
      hidden_size(hidden_size),
      seq_length(lookback_len),
      batch_first(batch_first) {
    
    // Ensure vectors are aligned for NEON operations
    lstm_layers.resize(num_layers);
    last_gradients.resize(num_layers);
    
    initialize_weights();
    reset_states();
}

void LSTMPredictor::reset_states() {
    h_state.clear();
    c_state.clear();
    
    // Align vectors for NEON
    const size_t alignment __attribute__((unused)) = 16; // 128-bit alignment for NEON
    const size_t padded_size = (hidden_size + 3) & ~3; // Round up to multiple of 4
    
    h_state.resize(num_layers);
    c_state.resize(num_layers);
    
    for (int i = 0; i < num_layers; ++i) {
        h_state[i].resize(padded_size, 0.0f);
        c_state[i].resize(padded_size, 0.0f);
        
        // Use NEON to zero out vectors
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (size_t j = 0; j < padded_size; j += 4) {
            vst1q_f32(&h_state[i][j], zero);
            vst1q_f32(&c_state[i][j], zero);
        }
    }
}

// NEON-optimized sigmoid implementation
inline float32x4_t sigmoid_neon(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t neg_x = vnegq_f32(x);
    float32x4_t exp_neg_x = vexpq_f32(neg_x);
    float32x4_t sum = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(one, sum);
}

float LSTMPredictor::sigmoid(float x) {
    // For single values, use scalar implementation
    return 1.0f / (1.0f + std::exp(-x));
}

// NEON-optimized tanh implementation
inline float32x4_t tanh_neon(float32x4_t x) {
    float32x4_t two = vdupq_n_f32(2.0f);
    
    // tanh(x) = 2sigmoid(2x) - 1
    float32x4_t two_x = vmulq_f32(two, x);
    float32x4_t sig = sigmoid_neon(two_x);
    float32x4_t two_sig = vmulq_f32(two, sig);
    float32x4_t one = vdupq_n_f32(1.0f);
    return vsubq_f32(two_sig, one);
}

float LSTMPredictor::tanh_custom(float x) {
    // For single values, use scalar implementation
    return std::tanh(x);
}

// Helper function to process vectors using NEON
void process_vector_neon(float* data, size_t size, float (*scalar_func)(float)) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 4 <= size; i += 4) {
        float32x4_t vec = vld1q_f32(data + i);
        float32x4_t result;
        
        // Use function pointer directly without class member comparison
        result = (scalar_func == static_cast<float(*)(float)>(std::tanh)) ? 
                 tanh_neon(vec) : sigmoid_neon(vec);
        
        vst1q_f32(data + i, result);
    }
    
    // Process remaining elements
    for (; i < size; ++i) {
        data[i] = scalar_func(data[i]);
    }
}

std::vector<float> LSTMPredictor::lstm_cell_forward(
    const std::vector<float>& input,
    std::vector<float>& h_state,
    std::vector<float>& c_state,
    const LSTMLayer& layer) {

    // Get the correct input size for this layer
    const int expected_layer_input = (current_layer == 0) ? input_size : hidden_size;
    const size_t aligned_input_size = (expected_layer_input + 3) & ~3;
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;

    // Debug output
    std::cout << "\nLSTM Cell Forward Debug (Layer " << current_layer << "):" << std::endl;
    std::cout << "Expected input size: " << expected_layer_input 
              << " (aligned: " << aligned_input_size << ")" << std::endl;
    std::cout << "Actual input size: " << input.size() << std::endl;
    std::cout << "Hidden size: " << hidden_size 
              << " (aligned: " << aligned_hidden_size << ")" << std::endl;
    std::cout << "Weight ih dimensions: " << layer.weight_ih.size() 
              << " x " << (layer.weight_ih.empty() ? 0 : layer.weight_ih[0].size()) << std::endl;
    std::cout << "Weight hh dimensions: " << layer.weight_hh.size() 
              << " x " << (layer.weight_hh.empty() ? 0 : layer.weight_hh[0].size()) << std::endl;
    std::cout << "H state size: " << h_state.size() << std::endl;
    std::cout << "C state size: " << c_state.size() << std::endl;

    // Verify input size
    if (input.size() != static_cast<size_t>(expected_layer_input)) {
        throw std::runtime_error(
            "Input size mismatch in lstm_cell_forward. Expected: " + 
            std::to_string(expected_layer_input) + ", Got: " + 
            std::to_string(input.size()) + " at layer " + 
            std::to_string(current_layer));
    }

    // Verify weight dimensions
    if (layer.weight_ih.size() != 4 * aligned_hidden_size) {
        throw std::runtime_error(
            "Weight ih rows mismatch. Expected: " + 
            std::to_string(4 * aligned_hidden_size) + ", Got: " + 
            std::to_string(layer.weight_ih.size()) + " at layer " + 
            std::to_string(current_layer));
    }

    if (!layer.weight_ih.empty() && layer.weight_ih[0].size() != aligned_input_size) {
        throw std::runtime_error(
            "Weight ih columns mismatch. Expected: " + 
            std::to_string(aligned_input_size) + ", Got: " + 
            std::to_string(layer.weight_ih[0].size()) + " at layer " + 
            std::to_string(current_layer) + 
            "\nInput size: " + std::to_string(input_size) + 
            "\nHidden size: " + std::to_string(hidden_size));
    }

    // Verify state dimensions and ensure alignment
    if (h_state.size() != static_cast<size_t>(aligned_hidden_size)) {
        std::cout << "Resizing h_state from " << h_state.size() 
                  << " to " << aligned_hidden_size << std::endl;
        h_state.resize(aligned_hidden_size, 0.0f);
    }
    if (c_state.size() != static_cast<size_t>(aligned_hidden_size)) {
        std::cout << "Resizing c_state from " << c_state.size() 
                  << " to " << aligned_hidden_size << std::endl;
        c_state.resize(aligned_hidden_size, 0.0f);
    }

    // Verify weight matrix dimensions
    if (layer.weight_ih.size() != 4 * hidden_size || 
        layer.weight_ih[0].size() != expected_layer_input) {
        throw std::runtime_error("Weight ih dimension mismatch");
    }
    if (layer.weight_hh.size() != 4 * hidden_size || 
        layer.weight_hh[0].size() != hidden_size) {
        throw std::runtime_error("Weight hh dimension mismatch");
    }
    
    // Declare cache_entry only if training_mode is true
    LSTMCacheEntry cache_entry;

    if (training_mode) {
        // Validate indices before accessing cache
        if (current_layer >= layer_cache.size() ||
            current_batch >= layer_cache[current_layer].size() ||
            current_timestep >= layer_cache[current_layer][current_batch].size()) {
            throw std::runtime_error("Invalid cache access");
        }
        
        cache_entry = layer_cache[current_layer][current_batch][current_timestep];
        cache_entry.input = input;
        
        // Initialize cache vectors with aligned size
        cache_entry.input_gate.resize(aligned_hidden_size, 0.0f);
        cache_entry.forget_gate.resize(aligned_hidden_size, 0.0f);
        cache_entry.cell_gate.resize(aligned_hidden_size, 0.0f);
        cache_entry.output_gate.resize(aligned_hidden_size, 0.0f);
        cache_entry.cell_state.resize(aligned_hidden_size, 0.0f);
        cache_entry.hidden_state.resize(aligned_hidden_size, 0.0f);
        cache_entry.prev_hidden = h_state;
        cache_entry.prev_cell = c_state;
    }
    
    // Initialize gates with biases using NEON
    std::vector<float> gates(4 * aligned_hidden_size, 0.0f);
    for (int h = 0; h < hidden_size; h += 4) {
        // Load bias vectors
        float32x4_t bias_ih_i = vld1q_f32(&layer.bias_ih[h]);
        float32x4_t bias_hh_i = vld1q_f32(&layer.bias_hh[h]);
        float32x4_t bias_ih_f = vld1q_f32(&layer.bias_ih[hidden_size + h]);
        float32x4_t bias_hh_f = vld1q_f32(&layer.bias_hh[hidden_size + h]);
        float32x4_t bias_ih_g = vld1q_f32(&layer.bias_ih[2 * hidden_size + h]);
        float32x4_t bias_hh_g = vld1q_f32(&layer.bias_hh[2 * hidden_size + h]);
        float32x4_t bias_ih_o = vld1q_f32(&layer.bias_ih[3 * hidden_size + h]);
        float32x4_t bias_hh_o = vld1q_f32(&layer.bias_hh[3 * hidden_size + h]);

        // Add biases
        vst1q_f32(&gates[h], vaddq_f32(bias_ih_i, bias_hh_i));
        vst1q_f32(&gates[hidden_size + h], vaddq_f32(bias_ih_f, bias_hh_f));
        vst1q_f32(&gates[2 * hidden_size + h], vaddq_f32(bias_ih_g, bias_hh_g));
        vst1q_f32(&gates[3 * hidden_size + h], vaddq_f32(bias_ih_o, bias_hh_o));
    }
    
    // Input to hidden contributions using NEON
    for (size_t i = 0; i < input.size(); ++i) {
        float32x4_t input_val = vdupq_n_f32(input[i]);
        
        for (int h = 0; h < hidden_size; h += 4) {
            // Load gate values and weights
            float32x4_t gates_i = vld1q_f32(&gates[h]);
            float32x4_t gates_f = vld1q_f32(&gates[hidden_size + h]);
            float32x4_t gates_g = vld1q_f32(&gates[2 * hidden_size + h]);
            float32x4_t gates_o = vld1q_f32(&gates[3 * hidden_size + h]);
            
            float32x4_t weight_i = vld1q_f32(&layer.weight_ih[h][i]);
            float32x4_t weight_f = vld1q_f32(&layer.weight_ih[hidden_size + h][i]);
            float32x4_t weight_g = vld1q_f32(&layer.weight_ih[2 * hidden_size + h][i]);
            float32x4_t weight_o = vld1q_f32(&layer.weight_ih[3 * hidden_size + h][i]);
            
            // Multiply-accumulate
            gates_i = vmlaq_f32(gates_i, weight_i, input_val);
            gates_f = vmlaq_f32(gates_f, weight_f, input_val);
            gates_g = vmlaq_f32(gates_g, weight_g, input_val);
            gates_o = vmlaq_f32(gates_o, weight_o, input_val);
            
            // Store results
            vst1q_f32(&gates[h], gates_i);
            vst1q_f32(&gates[hidden_size + h], gates_f);
            vst1q_f32(&gates[2 * hidden_size + h], gates_g);
            vst1q_f32(&gates[3 * hidden_size + h], gates_o);
        }
    }
    
    // Hidden to hidden contributions using NEON
    for (int h = 0; h < hidden_size; h += 4) {
        float32x4_t gates_i = vld1q_f32(&gates[h]);
        float32x4_t gates_f = vld1q_f32(&gates[hidden_size + h]);
        float32x4_t gates_g = vld1q_f32(&gates[2 * hidden_size + h]);
        float32x4_t gates_o = vld1q_f32(&gates[3 * hidden_size + h]);
        
        for (size_t i = 0; i < hidden_size; i += 4) {
            float32x4_t h_state_val = vld1q_f32(&h_state[i]);
            
            float32x4_t weight_i = vld1q_f32(&layer.weight_hh[h][i]);
            float32x4_t weight_f = vld1q_f32(&layer.weight_hh[hidden_size + h][i]);
            float32x4_t weight_g = vld1q_f32(&layer.weight_hh[2 * hidden_size + h][i]);
            float32x4_t weight_o = vld1q_f32(&layer.weight_hh[3 * hidden_size + h][i]);
            
            gates_i = vmlaq_f32(gates_i, weight_i, h_state_val);
            gates_f = vmlaq_f32(gates_f, weight_f, h_state_val);
            gates_g = vmlaq_f32(gates_g, weight_g, h_state_val);
            gates_o = vmlaq_f32(gates_o, weight_o, h_state_val);
        }
        
        vst1q_f32(&gates[h], gates_i);
        vst1q_f32(&gates[hidden_size + h], gates_f);
        vst1q_f32(&gates[2 * hidden_size + h], gates_g);
        vst1q_f32(&gates[3 * hidden_size + h], gates_o);
    }

    // Apply activations and update states using NEON
    for (int h = 0; h < hidden_size; h += 4) {
        // Load gate values
        float32x4_t i_vec = sigmoid_neon(vld1q_f32(&gates[h]));
        float32x4_t f_vec = sigmoid_neon(vld1q_f32(&gates[hidden_size + h]));
        float32x4_t g_vec = tanh_neon(vld1q_f32(&gates[2 * hidden_size + h]));
        float32x4_t o_vec = sigmoid_neon(vld1q_f32(&gates[3 * hidden_size + h]));
        
        // Load current cell state
        float32x4_t c_state_vec = vld1q_f32(&c_state[h]);
        
        // Compute new cell state
        float32x4_t new_cell = vaddq_f32(
            vmulq_f32(f_vec, c_state_vec),
            vmulq_f32(i_vec, g_vec)
        );
        
        // Compute new hidden state
        float32x4_t new_hidden = vmulq_f32(o_vec, tanh_neon(new_cell));
        
        // Store results
        vst1q_f32(&c_state[h], new_cell);
        vst1q_f32(&h_state[h], new_hidden);
        
        if (training_mode) {
            vst1q_f32(&cache_entry.input_gate[h], i_vec);
            vst1q_f32(&cache_entry.forget_gate[h], f_vec);
            vst1q_f32(&cache_entry.cell_gate[h], g_vec);
            vst1q_f32(&cache_entry.output_gate[h], o_vec);
            vst1q_f32(&cache_entry.cell_state[h], new_cell);
            vst1q_f32(&cache_entry.hidden_state[h], new_hidden);
        }
    }

    // Create aligned output vector
    std::vector<float> output(hidden_size);
    std::memcpy(output.data(), h_state.data(), hidden_size * sizeof(float));

    // Store cache if in training mode
    if (training_mode) {
        layer_cache[current_layer][current_batch][current_timestep] = cache_entry;
    }

    return output;
}

LSTMPredictor::LSTMOutput LSTMPredictor::forward(
    const std::vector<std::vector<std::vector<float>>>& x,
    const std::vector<std::vector<float>>* initial_hidden,
    const std::vector<std::vector<float>>* initial_cell) {
    
    // Input validation
    for (size_t batch = 0; batch < x.size(); ++batch) {
        for (size_t seq = 0; seq < x[batch].size(); ++seq) {
            if (x[batch][seq].size() != input_size) {
                throw std::runtime_error("Input dimension mismatch in sequence");
            }
        }
    }
    
    try {
        size_t batch_size = x.size();
        size_t seq_len = x[0].size();
        
        // Calculate aligned sizes for NEON operations
        const size_t aligned_hidden_size = (hidden_size + 3) & ~3; // Round up to multiple of 4
        const size_t aligned_input_size = (input_size + 3) & ~3;
        
        // Initialize layer cache for training with aligned sizes
        if (training_mode) {
            layer_cache.clear();
            layer_cache.resize(num_layers);
            
            for (int layer = 0; layer < num_layers; ++layer) {
                layer_cache[layer].resize(batch_size);
                for (size_t batch = 0; batch < batch_size; ++batch) {
                    layer_cache[layer][batch].resize(seq_len);
                }
            }
        }
        
        // Initialize output structure with aligned sizes
        LSTMOutput output;
        output.sequence_output.resize(batch_size, 
            std::vector<std::vector<float>>(seq_len, 
                std::vector<float>(hidden_size)));  // Note: Not aligned size here
        
        // Initialize or use provided states
        if (!initial_hidden || !initial_cell) {
            h_state.resize(num_layers);
            c_state.resize(num_layers);
            
            for (int layer = 0; layer < num_layers; ++layer) {
                h_state[layer].resize(hidden_size, 0.0f);  // Use actual size
                c_state[layer].resize(hidden_size, 0.0f);  // Use actual size
            }
        } else {
            h_state = *initial_hidden;
            c_state = *initial_cell;
        }
        
        // Process each batch and timestep
        for (size_t batch = 0; batch < batch_size; ++batch) {
            current_batch = batch;
            
            if (!initial_hidden || !initial_cell) {
                reset_states();
            }
            
            for (size_t t = 0; t < seq_len; ++t) {
                current_timestep = t;
                std::vector<float> layer_input = x[batch][t];
                
                // Process through LSTM layers
                for (int layer = 0; layer < num_layers; ++layer) {
                    current_layer = layer;
                    layer_input = lstm_cell_forward(
                        layer_input,
                        h_state[layer],
                        c_state[layer],
                        lstm_layers[layer]
                    );
                }
                
                // Store output
                output.sequence_output[batch][t] = layer_input;
            }
        }
        
        // Store final states
        output.final_hidden = h_state;
        output.final_cell = c_state;
        
        return output;
        
    } catch (const std::exception& e) {
        throw;
    }
}

void LSTMPredictor::set_lstm_weights(int layer, 
                                   const std::vector<std::vector<float>>& w_ih,
                                   const std::vector<std::vector<float>>& w_hh) {
    if (layer >= num_layers) {
        return;
    }

    // Calculate aligned sizes
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;  // Round up to multiple of 4
    const size_t aligned_input_size = (input_size + 3) & ~3;

    // Resize weight matrices with aligned dimensions
    lstm_layers[layer].weight_ih.resize(4 * aligned_hidden_size);
    lstm_layers[layer].weight_hh.resize(4 * aligned_hidden_size);

    for (size_t i = 0; i < 4 * hidden_size; ++i) {
        lstm_layers[layer].weight_ih[i].resize(aligned_input_size, 0.0f);
        lstm_layers[layer].weight_hh[i].resize(aligned_hidden_size, 0.0f);

        // Copy input weights with proper alignment
        std::memcpy(lstm_layers[layer].weight_ih[i].data(),
                   w_ih[i].data(),
                   std::min(w_ih[i].size(), aligned_input_size) * sizeof(float));

        // Copy hidden weights with proper alignment
        std::memcpy(lstm_layers[layer].weight_hh[i].data(),
                   w_hh[i].data(),
                   std::min(w_hh[i].size(), aligned_hidden_size) * sizeof(float));
    }
}

void LSTMPredictor::set_lstm_bias(int layer,
                                 const std::vector<float>& b_ih,
                                 const std::vector<float>& b_hh) {
    if (layer >= num_layers) {
        return;
    }

    // Calculate aligned size
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;  // Round up to multiple of 4
    
    // Resize bias vectors with aligned size
    lstm_layers[layer].bias_ih.resize(4 * aligned_hidden_size, 0.0f);
    lstm_layers[layer].bias_hh.resize(4 * aligned_hidden_size, 0.0f);

    // Copy biases with proper alignment
    std::memcpy(lstm_layers[layer].bias_ih.data(),
                b_ih.data(),
                std::min(b_ih.size(), 4 * aligned_hidden_size) * sizeof(float));
    
    std::memcpy(lstm_layers[layer].bias_hh.data(),
                b_hh.data(),
                std::min(b_hh.size(), 4 * aligned_hidden_size) * sizeof(float));
}

void LSTMPredictor::set_fc_weights(const std::vector<std::vector<float>>& weights,
                                  const std::vector<float>& bias) {
    // Calculate aligned sizes
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_output_size = (num_classes + 3) & ~3;

    // Resize fc_weight with aligned dimensions
    fc_weight.resize(aligned_output_size);
    for (auto& row : fc_weight) {
        row.resize(aligned_hidden_size, 0.0f);
    }

    // Copy weights with proper alignment
    for (size_t i = 0; i < std::min(weights.size(), fc_weight.size()); ++i) {
        std::memcpy(fc_weight[i].data(),
                   weights[i].data(),
                   std::min(weights[i].size(), aligned_hidden_size) * sizeof(float));
    }

    // Resize and copy fc_bias with aligned size
    fc_bias.resize(aligned_output_size, 0.0f);
    std::memcpy(fc_bias.data(),
                bias.data(),
                std::min(bias.size(), aligned_output_size) * sizeof(float));
}

void LSTMPredictor::backward_linear_layer(
    const std::vector<float>& grad_output,
    const std::vector<float>& last_hidden,
    std::vector<std::vector<float>>& weight_grad,
    std::vector<float>& bias_grad,
    std::vector<float>& input_grad) {
    
    // Check dimensions
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
    
    // Calculate aligned sizes
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_num_classes = (num_classes + 3) & ~3;
    
    // Initialize gradients with aligned dimensions
    weight_grad.resize(aligned_num_classes);
    for (auto& row : weight_grad) {
        row.resize(aligned_hidden_size, 0.0f);
    }
    
    // Copy and align grad_output for bias
    bias_grad.resize(aligned_num_classes, 0.0f);
    std::memcpy(bias_grad.data(), grad_output.data(), 
                num_classes * sizeof(float));
    
    // Align input gradients
    input_grad.resize(aligned_hidden_size, 0.0f);
    
    // Create aligned copies of input data
    std::vector<float> aligned_grad_output(aligned_num_classes, 0.0f);
    std::vector<float> aligned_last_hidden(aligned_hidden_size, 0.0f);
    std::memcpy(aligned_grad_output.data(), grad_output.data(), 
                num_classes * sizeof(float));
    std::memcpy(aligned_last_hidden.data(), last_hidden.data(), 
                hidden_size * sizeof(float));
    
    // Compute weight gradients using NEON
    for (int i = 0; i < num_classes; i += 4) {
        float32x4_t grad_vec = vld1q_f32(&aligned_grad_output[i]);
        
        for (int j = 0; j < hidden_size; j += 4) {
            float32x4_t hidden_vec = vld1q_f32(&aligned_last_hidden[j]);
            
            // Use individual lanes with constant indices
            float32x4_t result0 = vmulq_n_f32(hidden_vec, vgetq_lane_f32(grad_vec, 0));
            if (i + 1 < num_classes) {
                float32x4_t result1 = vmulq_n_f32(hidden_vec, vgetq_lane_f32(grad_vec, 1));
                vst1q_f32(&weight_grad[i + 1][j], result1);
            }
            if (i + 2 < num_classes) {
                float32x4_t result2 = vmulq_n_f32(hidden_vec, vgetq_lane_f32(grad_vec, 2));
                vst1q_f32(&weight_grad[i + 2][j], result2);
            }
            if (i + 3 < num_classes) {
                float32x4_t result3 = vmulq_n_f32(hidden_vec, vgetq_lane_f32(grad_vec, 3));
                vst1q_f32(&weight_grad[i + 3][j], result3);
            }
            vst1q_f32(&weight_grad[i][j], result0);
        }
    }
    
    // Compute input gradients using NEON
    for (int i = 0; i < hidden_size; i += 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        
        // Process 4 elements at a time for each class
        for (int j = 0; j < num_classes; j += 4) {
            float32x4_t grad_vec = vld1q_f32(&aligned_grad_output[j]);
            
            // Load 4x4 block of weights
            float32x4_t w0 = vld1q_f32(&fc_weight[j][i]);
            float32x4_t w1 = vld1q_f32(&fc_weight[j + 1][i]);
            float32x4_t w2 = vld1q_f32(&fc_weight[j + 2][i]);
            float32x4_t w3 = vld1q_f32(&fc_weight[j + 3][i]);
            
            // Multiply-accumulate for each row
            sum = vmlaq_n_f32(sum, w0, vgetq_lane_f32(grad_vec, 0));
            if (j + 1 < num_classes) sum = vmlaq_n_f32(sum, w1, vgetq_lane_f32(grad_vec, 1));
            if (j + 2 < num_classes) sum = vmlaq_n_f32(sum, w2, vgetq_lane_f32(grad_vec, 2));
            if (j + 3 < num_classes) sum = vmlaq_n_f32(sum, w3, vgetq_lane_f32(grad_vec, 3));
        }
        
        // Store the accumulated result
        vst1q_f32(&input_grad[i], sum);
    }
    
    // Handle remaining elements (if any)
    for (int i = (hidden_size / 4) * 4; i < hidden_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum += fc_weight[j][i] * grad_output[j];
        }
        input_grad[i] = sum;
    }
    
    // Trim output vectors back to original sizes if needed
    for (auto& row : weight_grad) {
        row.resize(hidden_size);
    }
    weight_grad.resize(num_classes);
    bias_grad.resize(num_classes);
    input_grad.resize(hidden_size);
}

std::vector<LSTMPredictor::LSTMGradients> LSTMPredictor::backward_lstm_layer(
    const std::vector<float>& grad_output,
    const std::vector<std::vector<std::vector<LSTMCacheEntry>>>& cache,
    float learning_rate) {
    
    // Dimension validation
    if (grad_output.size() != hidden_size) {
        throw std::runtime_error("grad_output size mismatch in backward_lstm_layer");
    }
    if (cache.size() != num_layers) {
        throw std::runtime_error("cache layer count mismatch in backward_lstm_layer");
    }
    
    // Calculate aligned sizes
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_input_size = (input_size + 3) & ~3;
    
    std::vector<LSTMGradients> layer_grads(num_layers);
    
    // Initialize gradients with aligned sizes
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        int aligned_input_size_layer = (layer == 0) ? aligned_input_size : aligned_hidden_size;
        
        layer_grads[layer].weight_ih_grad.resize(4 * aligned_hidden_size);
        layer_grads[layer].weight_hh_grad.resize(4 * aligned_hidden_size);
        
        for (auto& row : layer_grads[layer].weight_ih_grad) {
            row.resize(aligned_input_size_layer, 0.0f);
        }
        for (auto& row : layer_grads[layer].weight_hh_grad) {
            row.resize(aligned_hidden_size, 0.0f);
        }
        
        layer_grads[layer].bias_ih_grad.resize(4 * aligned_hidden_size, 0.0f);
        layer_grads[layer].bias_hh_grad.resize(4 * aligned_hidden_size, 0.0f);
    }
    
    // Initialize aligned dh_next and dc_next
    std::vector<std::vector<float>> dh_next(num_layers, std::vector<float>(aligned_hidden_size, 0.0f));
    std::vector<std::vector<float>> dc_next(num_layers, std::vector<float>(aligned_hidden_size, 0.0f));
    
    // Process each layer backwards
    for (int layer = num_layers - 1; layer >= 0; --layer) {
        if (current_batch >= cache[layer].size()) {
            throw std::runtime_error("Cache batch index out of bounds");
        }
        
        std::vector<float> dh = dh_next[layer];
        std::vector<float> dc = dc_next[layer];
        const auto& layer_cache = cache[layer][current_batch];
        
        // Add grad_output for last layer using NEON
        if (layer == num_layers - 1) {
            for (int h = 0; h < hidden_size; h += 4) {
                float32x4_t dh_vec = vld1q_f32(&dh[h]);
                float32x4_t grad_vec = vld1q_f32(&grad_output[h]);
                vst1q_f32(&dh[h], vaddq_f32(dh_vec, grad_vec));
            }
        }

        // Process timesteps in reverse
        for (int t = layer_cache.size() - 1; t >= 0; --t) {
            const auto& cache_entry = layer_cache[t];
            std::vector<float> dh_prev(aligned_hidden_size, 0.0f);
            std::vector<float> dc_prev(aligned_hidden_size, 0.0f);

            // Process hidden units using NEON
            for (int h = 0; h < hidden_size; h += 4) {
                // Load vectors
                float32x4_t c_state = vld1q_f32(&cache_entry.cell_state[h]);
                float32x4_t tanh_c = tanh_neon(c_state);
                float32x4_t dho = vld1q_f32(&dh[h]);
                float32x4_t o_gate = vld1q_f32(&cache_entry.output_gate[h]);
                
                // Calculate dc_t
                float32x4_t one = vdupq_n_f32(1.0f);
                float32x4_t tanh_grad = vsubq_f32(one, vmulq_f32(tanh_c, tanh_c));
                float32x4_t dc_t = vmulq_f32(dho, vmulq_f32(o_gate, tanh_grad));
                float32x4_t dc_prev_t = vld1q_f32(&dc[h]);
                dc_t = vaddq_f32(dc_t, dc_prev_t);
                
                // Calculate gate gradients
                float32x4_t i_gate = vld1q_f32(&cache_entry.input_gate[h]);
                float32x4_t f_gate = vld1q_f32(&cache_entry.forget_gate[h]);
                float32x4_t g_gate = vld1q_f32(&cache_entry.cell_gate[h]);
                
                float32x4_t do_t = vmulq_f32(vmulq_f32(dho, tanh_c),
                    vmulq_f32(o_gate, vsubq_f32(one, o_gate)));
                    
                float32x4_t di_t = vmulq_f32(vmulq_f32(dc_t, g_gate),
                    vmulq_f32(i_gate, vsubq_f32(one, i_gate)));
                    
                float32x4_t df_t = vmulq_f32(vmulq_f32(dc_t, 
                    vld1q_f32(&cache_entry.prev_cell[h])),
                    vmulq_f32(f_gate, vsubq_f32(one, f_gate)));
                    
                float32x4_t dg_t = vmulq_f32(vmulq_f32(dc_t, i_gate),
                    vsubq_f32(one, vmulq_f32(g_gate, g_gate)));

                // Process gates with constant indices
                for (int gate = 0; gate < 4; ++gate) {
                    float32x4_t gate_grad;
                    switch(gate) {
                        case 0: gate_grad = di_t; break;
                        case 1: gate_grad = df_t; break;
                        case 2: gate_grad = dg_t; break;
                        case 3: gate_grad = do_t; break;
                    }
                    
                    // Use separate operations for each lane
                    float gate_val0 = vgetq_lane_f32(gate_grad, 0);
                    float gate_val1 = vgetq_lane_f32(gate_grad, 1);
                    float gate_val2 = vgetq_lane_f32(gate_grad, 2);
                    float gate_val3 = vgetq_lane_f32(gate_grad, 3);
                    
                    float32x4_t h_prev = vld1q_f32(&cache_entry.prev_hidden[h]);
                    float32x4_t dh_prev_vec = vld1q_f32(&dh_prev[h]);
                    
                    // Update gradients for each lane separately
                    if (h + 0 < hidden_size) {
                        float32x4_t curr_grad = vld1q_f32(
                            &layer_grads[layer].weight_ih_grad[gate * hidden_size + h][0]);
                        curr_grad = vmlaq_n_f32(curr_grad, h_prev, gate_val0);
                        vst1q_f32(&layer_grads[layer].weight_ih_grad[gate * hidden_size + h][0],
                                 curr_grad);
                        
                        float32x4_t weight_vec = vld1q_f32(
                            &lstm_layers[layer].weight_ih[gate * hidden_size + h][0]);
                        dh_prev_vec = vmlaq_n_f32(dh_prev_vec, weight_vec, gate_val0);
                    }
                    
                    if (h + 1 < hidden_size) {
                        float32x4_t curr_grad = vld1q_f32(
                            &layer_grads[layer].weight_ih_grad[gate * hidden_size + h + 1][0]);
                        curr_grad = vmlaq_n_f32(curr_grad, h_prev, gate_val1);
                        vst1q_f32(&layer_grads[layer].weight_ih_grad[gate * hidden_size + h + 1][0],
                                 curr_grad);
                        
                        float32x4_t weight_vec = vld1q_f32(
                            &lstm_layers[layer].weight_ih[gate * hidden_size + h + 1][0]);
                        dh_prev_vec = vmlaq_n_f32(dh_prev_vec, weight_vec, gate_val1);
                    }
                    
                    if (h + 2 < hidden_size) {
                        float32x4_t curr_grad = vld1q_f32(
                            &layer_grads[layer].weight_ih_grad[gate * hidden_size + h + 2][0]);
                        curr_grad = vmlaq_n_f32(curr_grad, h_prev, gate_val2);
                        vst1q_f32(&layer_grads[layer].weight_ih_grad[gate * hidden_size + h + 2][0],
                                 curr_grad);
                        
                        float32x4_t weight_vec = vld1q_f32(
                            &lstm_layers[layer].weight_ih[gate * hidden_size + h + 2][0]);
                        dh_prev_vec = vmlaq_n_f32(dh_prev_vec, weight_vec, gate_val2);
                    }
                    
                    if (h + 3 < hidden_size) {
                        float32x4_t curr_grad = vld1q_f32(
                            &layer_grads[layer].weight_ih_grad[gate * hidden_size + h + 3][0]);
                        curr_grad = vmlaq_n_f32(curr_grad, h_prev, gate_val3);
                        vst1q_f32(&layer_grads[layer].weight_ih_grad[gate * hidden_size + h + 3][0],
                                 curr_grad);
                        
                        float32x4_t weight_vec = vld1q_f32(
                            &lstm_layers[layer].weight_ih[gate * hidden_size + h + 3][0]);
                        dh_prev_vec = vmlaq_n_f32(dh_prev_vec, weight_vec, gate_val3);
                    }
                    
                    vst1q_f32(&dh_prev[h], dh_prev_vec);
                }
                
                // Accumulate bias gradients
                vst1q_f32(&layer_grads[layer].bias_ih_grad[h], di_t);
                vst1q_f32(&layer_grads[layer].bias_ih_grad[hidden_size + h], df_t);
                vst1q_f32(&layer_grads[layer].bias_ih_grad[2 * hidden_size + h], dg_t);
                vst1q_f32(&layer_grads[layer].bias_ih_grad[3 * hidden_size + h], do_t);
                
                // Calculate cell state gradient for previous timestep
                float32x4_t dc_prev_vec = vmulq_f32(dc_t, f_gate);
                vst1q_f32(&dc_prev[h], dc_prev_vec);
            }
            
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

float LSTMPredictor::compute_loss(const std::vector<float>& output,
                                const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::runtime_error("Output and target size mismatch in compute_loss");
    }
    
    const size_t aligned_size = (output.size() + 3) & ~3; // Round up to multiple of 4
    float32x4_t loss_vec = vdupq_n_f32(0.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    
    // Process 4 elements at a time using NEON
    size_t i = 0;
    for (; i + 4 <= output.size(); i += 4) {
        float32x4_t output_vec = vld1q_f32(&output[i]);
        float32x4_t target_vec = vld1q_f32(&target[i]);
        float32x4_t diff = vsubq_f32(output_vec, target_vec);
        float32x4_t squared = vmulq_f32(diff, diff);
        loss_vec = vaddq_f32(loss_vec, vmulq_f32(half, squared));
    }
    
    // Sum up the vector elements
    float loss = vgetq_lane_f32(loss_vec, 0) + 
                 vgetq_lane_f32(loss_vec, 1) + 
                 vgetq_lane_f32(loss_vec, 2) + 
                 vgetq_lane_f32(loss_vec, 3);
    
    // Handle remaining elements
    for (; i < output.size(); ++i) {
        float diff = output[i] - target[i];
        loss += 0.5f * diff * diff;
    }
    
    return loss;
}

std::vector<float> LSTMPredictor::get_final_prediction(const LSTMOutput& lstm_output) {
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_num_classes = (num_classes + 3) & ~3;
    
    std::vector<float> final_output(aligned_num_classes, 0.0f);
    const auto& final_hidden = lstm_output.sequence_output.back().back();
    
    // Copy bias values using NEON
    for (int i = 0; i < num_classes; i += 4) {
        float32x4_t bias_vec = vld1q_f32(&fc_bias[i]);
        vst1q_f32(&final_output[i], bias_vec);
    }
    
    // Modify matrix multiplication
    for (int i = 0; i < num_classes; i += 4) {
        float32x4_t sum_vec = vld1q_f32(&final_output[i]);
        
        for (int j = 0; j < hidden_size; j += 4) {
            float32x4_t hidden_vec = vld1q_f32(&final_hidden[j]);
            
            // Use constant indices
            float32x4_t weight_vec0 = vld1q_f32(&fc_weight[i][j]);
            sum_vec = vmlaq_n_f32(sum_vec, weight_vec0, vgetq_lane_f32(hidden_vec, 0));
            
            if (j + 1 < hidden_size) {
                float32x4_t weight_vec1 = vld1q_f32(&fc_weight[i][j + 1]);
                sum_vec = vmlaq_n_f32(sum_vec, weight_vec1, vgetq_lane_f32(hidden_vec, 1));
            }
            if (j + 2 < hidden_size) {
                float32x4_t weight_vec2 = vld1q_f32(&fc_weight[i][j + 2]);
                sum_vec = vmlaq_n_f32(sum_vec, weight_vec2, vgetq_lane_f32(hidden_vec, 2));
            }
            if (j + 3 < hidden_size) {
                float32x4_t weight_vec3 = vld1q_f32(&fc_weight[i][j + 3]);
                sum_vec = vmlaq_n_f32(sum_vec, weight_vec3, vgetq_lane_f32(hidden_vec, 3));
            }
        }
        
        vst1q_f32(&final_output[i], sum_vec);
    }
    
    // Resize to actual output size
    final_output.resize(num_classes);
    return final_output;
}

void LSTMPredictor::initialize_weights() {
    float k = 1.0f / std::sqrt(hidden_size);
    std::uniform_real_distribution<float> dist(-k, k);
    std::mt19937 gen(random_seed);
    
    // Calculate aligned sizes
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_input_size = (input_size + 3) & ~3;
    
    // Initialize FC layer first with original dimensions
    fc_weight.resize(num_classes, std::vector<float>(hidden_size));
    fc_bias.resize(num_classes);
    
    // Initialize FC weights using NEON
    for (int i = 0; i < num_classes; i += 4) {
        // Initialize bias
        float32x4_t rand_bias = {dist(gen), dist(gen), dist(gen), dist(gen)};
        if (i + 4 <= num_classes) {
            vst1q_f32(&fc_bias[i], rand_bias);
        } else {
            float temp[4];
            vst1q_f32(temp, rand_bias);
            for (int k = 0; k < num_classes - i; ++k) {
                fc_bias[i + k] = temp[k];
            }
        }
        
        // Initialize weights
        for (int j = 0; j < hidden_size; j += 4) {
            float32x4_t rand_vec = {dist(gen), dist(gen), dist(gen), dist(gen)};
            for (int k = 0; k < 4 && i + k < num_classes; ++k) {
                if (j + 4 <= hidden_size) {
                    vst1q_f32(&fc_weight[i + k][j], rand_vec);
                } else {
                    float temp[4];
                    vst1q_f32(temp, rand_vec);
                    for (int l = 0; l < hidden_size - j; ++l) {
                        fc_weight[i + k][j + l] = temp[l];
                    }
                }
            }
        }
    }

    // Initialize LSTM layers with aligned dimensions
    lstm_layers.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        const size_t aligned_input_size_layer = 
            (layer == 0) ? aligned_input_size : aligned_hidden_size;
        
        // Initialize with aligned dimensions
        lstm_layers[layer].weight_ih.resize(4 * hidden_size, 
            std::vector<float>(aligned_input_size_layer, 0.0f));
        lstm_layers[layer].weight_hh.resize(4 * hidden_size, 
            std::vector<float>(aligned_hidden_size, 0.0f));
        lstm_layers[layer].bias_ih.resize(4 * hidden_size);
        lstm_layers[layer].bias_hh.resize(4 * hidden_size);
        
        // Initialize weights and biases using NEON
        for (int i = 0; i < 4 * hidden_size; i += 4) {
            // Initialize biases
            float32x4_t rand_bias_ih = {dist(gen), dist(gen), dist(gen), dist(gen)};
            float32x4_t rand_bias_hh = {dist(gen), dist(gen), dist(gen), dist(gen)};
            if (i + 4 <= 4 * hidden_size) {
                vst1q_f32(&lstm_layers[layer].bias_ih[i], rand_bias_ih);
                vst1q_f32(&lstm_layers[layer].bias_hh[i], rand_bias_hh);
            } else {
                float temp_ih[4], temp_hh[4];
                vst1q_f32(temp_ih, rand_bias_ih);
                vst1q_f32(temp_hh, rand_bias_hh);
                for (int k = 0; k < 4 * hidden_size - i; ++k) {
                    lstm_layers[layer].bias_ih[i + k] = temp_ih[k];
                    lstm_layers[layer].bias_hh[i + k] = temp_hh[k];
                }
            }
            
            // Initialize weights with actual values only up to input_size_layer
            for (int j = 0; j < aligned_input_size_layer; j += 4) {
                float32x4_t rand_vec = {dist(gen), dist(gen), dist(gen), dist(gen)};
                for (int k = 0; k < 4 && i + k < 4 * hidden_size; ++k) {
                    if (j + 4 <= aligned_input_size_layer) {
                        vst1q_f32(&lstm_layers[layer].weight_ih[i + k][j], rand_vec);
                    } else {
                        float temp[4];
                        vst1q_f32(temp, rand_vec);
                        for (int l = 0; l < aligned_input_size_layer - j; ++l) {
                            lstm_layers[layer].weight_ih[i + k][j + l] = temp[l];
                        }
                    }
                }
            }
            
            // Initialize weights with actual values only up to hidden_size
            for (int j = 0; j < aligned_hidden_size; j += 4) {
                float32x4_t rand_vec = {dist(gen), dist(gen), dist(gen), dist(gen)};
                for (int k = 0; k < 4 && i + k < 4 * hidden_size; ++k) {
                    if (j + 4 <= aligned_hidden_size) {
                        vst1q_f32(&lstm_layers[layer].weight_hh[i + k][j], rand_vec);
                    } else {
                        float temp[4];
                        vst1q_f32(temp, rand_vec);
                        for (int l = 0; l < aligned_hidden_size - j; ++l) {
                            lstm_layers[layer].weight_hh[i + k][j + l] = temp[l];
                        }
                    }
                }
            }
        }
    }
}

void LSTMPredictor::initialize_adam_states() {
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_input_size = (input_size + 3) & ~3;
    const size_t aligned_num_classes = (num_classes + 3) & ~3;
    
    try {
        // Initialize FC layer vectors with aligned sizes
        m_fc_weight.resize(aligned_num_classes);
        v_fc_weight.resize(aligned_num_classes);
        for (auto& row : m_fc_weight) row.resize(aligned_hidden_size, 0.0f);
        for (auto& row : v_fc_weight) row.resize(aligned_hidden_size, 0.0f);
        
        m_fc_bias.resize(aligned_num_classes, 0.0f);
        v_fc_bias.resize(aligned_num_classes, 0.0f);
        
        // Initialize zero vector for NEON operations
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        
        // Initialize LSTM layer states with correct dimensions
        m_weight_ih.resize(num_layers);
        v_weight_ih.resize(num_layers);
        m_weight_hh.resize(num_layers);
        v_weight_hh.resize(num_layers);
        m_bias_ih.resize(num_layers);
        v_bias_ih.resize(num_layers);
        m_bias_hh.resize(num_layers);
        v_bias_hh.resize(num_layers);
        
        for (int layer = 0; layer < num_layers; ++layer) {
            int aligned_input_size_layer = (layer == 0) ? aligned_input_size : aligned_hidden_size;
            
            // Initialize weight matrices with correct dimensions
            m_weight_ih[layer].resize(4 * aligned_hidden_size);
            v_weight_ih[layer].resize(4 * aligned_hidden_size);
            m_weight_hh[layer].resize(4 * aligned_hidden_size);
            v_weight_hh[layer].resize(4 * aligned_hidden_size);
            
            // Resize each row correctly
            for (auto& row : m_weight_ih[layer]) row.resize(aligned_input_size_layer, 0.0f);
            for (auto& row : v_weight_ih[layer]) row.resize(aligned_input_size_layer, 0.0f);
            for (auto& row : m_weight_hh[layer]) row.resize(aligned_hidden_size, 0.0f);
            for (auto& row : v_weight_hh[layer]) row.resize(aligned_hidden_size, 0.0f);
            
            // Zero out matrices using NEON
            for (int i = 0; i < 4 * aligned_hidden_size; i += 4) {
                for (int j = 0; j < aligned_input_size_layer; j += 4) {
                    vst1q_f32(&m_weight_ih[layer][i][j], zero_vec);
                    vst1q_f32(&v_weight_ih[layer][i][j], zero_vec);
                }
                for (int j = 0; j < aligned_hidden_size; j += 4) {
                    vst1q_f32(&m_weight_hh[layer][i][j], zero_vec);
                    vst1q_f32(&v_weight_hh[layer][i][j], zero_vec);
                }
            }
            
            // Initialize and zero out bias vectors
            m_bias_ih[layer].resize(4 * aligned_hidden_size, 0.0f);
            v_bias_ih[layer].resize(4 * aligned_hidden_size, 0.0f);
            m_bias_hh[layer].resize(4 * aligned_hidden_size, 0.0f);
            v_bias_hh[layer].resize(4 * aligned_hidden_size, 0.0f);
            
            // Zero out bias vectors using NEON
            for (int i = 0; i < 4 * aligned_hidden_size; i += 4) {
                vst1q_f32(&m_bias_ih[layer][i], zero_vec);
                vst1q_f32(&v_bias_ih[layer][i], zero_vec);
                vst1q_f32(&m_bias_hh[layer][i], zero_vec);
                vst1q_f32(&v_bias_hh[layer][i], zero_vec);
            }
        }
        
        adam_initialized = true;
        
    } catch (const std::exception& e) {
        throw;
    }
}

void LSTMPredictor::apply_adam_update(
    std::vector<std::vector<float>>& weights, 
    std::vector<std::vector<float>>& grads,
    std::vector<std::vector<float>>& m_t, 
    std::vector<std::vector<float>>& v_t,
    float learning_rate, float beta1, float beta2, float epsilon, int t) {
    
    // Validation checks
    if (weights.empty() || grads.empty() || m_t.empty() || v_t.empty() ||
        weights[0].empty() || grads[0].empty() || m_t[0].empty() || v_t[0].empty() ||
        weights.size() != grads.size() || weights[0].size() != grads[0].size() ||
        weights.size() != m_t.size() || weights[0].size() != m_t[0].size() ||
        weights.size() != v_t.size() || weights[0].size() != v_t[0].size() ||
        t <= 0) {
        throw std::runtime_error("Invalid inputs in Adam update");
    }
    
    // Prepare NEON constants
    float32x4_t beta1_vec = vdupq_n_f32(beta1);
    float32x4_t one_minus_beta1 = vdupq_n_f32(1.0f - beta1);
    float32x4_t beta2_vec = vdupq_n_f32(beta2);
    float32x4_t one_minus_beta2 = vdupq_n_f32(1.0f - beta2);
    float32x4_t eps_vec = vdupq_n_f32(epsilon);
    float32x4_t lr_vec = vdupq_n_f32(learning_rate);
    
    float beta1_correction = 1.0f - pow_float(beta1, static_cast<float>(t));
    float beta2_correction = 1.0f - pow_float(beta2, static_cast<float>(t));
    float32x4_t beta1_corr_vec = vdupq_n_f32(beta1_correction);
    float32x4_t beta2_corr_vec = vdupq_n_f32(beta2_correction);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); j += 4) {
            // Load vectors
            float32x4_t grad_vec = vld1q_f32(&grads[i][j]);
            float32x4_t m_vec = vld1q_f32(&m_t[i][j]);
            float32x4_t v_vec = vld1q_f32(&v_t[i][j]);
            
            // Update biased first moment
            float32x4_t m_new = vmlaq_f32(
                vmulq_f32(one_minus_beta1, grad_vec),
                beta1_vec, m_vec
            );
            
            // Update biased second moment
            float32x4_t grad_squared = vmulq_f32(grad_vec, grad_vec);
            float32x4_t v_new = vmlaq_f32(
                vmulq_f32(one_minus_beta2, grad_squared),
                beta2_vec, v_vec
            );
            
            // Compute bias-corrected moments
            float32x4_t m_hat = vdivq_f32(m_new, beta1_corr_vec);
            float32x4_t v_hat = vdivq_f32(v_new, beta2_corr_vec);
            
            // Compute update
            float32x4_t denom = vaddq_f32(vsqrtq_f32(v_hat), eps_vec);
            float32x4_t update = vdivq_f32(m_hat, denom);
            
            // Update weights
            float32x4_t weight_vec = vld1q_f32(&weights[i][j]);
            weight_vec = vsubq_f32(weight_vec, vmulq_f32(lr_vec, update));
            
            // Store results
            vst1q_f32(&weights[i][j], weight_vec);
            vst1q_f32(&m_t[i][j], m_new);
            vst1q_f32(&v_t[i][j], v_new);
        }
    }
}

void LSTMPredictor::apply_adam_update(
    std::vector<float>& biases, 
    std::vector<float>& grads,
    std::vector<float>& m_t, 
    std::vector<float>& v_t,
    float learning_rate, float beta1, float beta2, float epsilon, int t) {
    
    if (t <= 0) {
        throw std::invalid_argument("Adam timestep must be positive");
    }
    
    // Prepare NEON constants
    float32x4_t beta1_vec = vdupq_n_f32(beta1);
    float32x4_t one_minus_beta1 = vdupq_n_f32(1.0f - beta1);
    float32x4_t beta2_vec = vdupq_n_f32(beta2);
    float32x4_t one_minus_beta2 = vdupq_n_f32(1.0f - beta2);
    float32x4_t eps_vec = vdupq_n_f32(epsilon);
    float32x4_t lr_vec = vdupq_n_f32(learning_rate);
    
    float beta1_correction = 1.0f - pow_float(beta1, static_cast<float>(t));
    float beta2_correction = 1.0f - pow_float(beta2, static_cast<float>(t));
    float32x4_t beta1_corr_vec = vdupq_n_f32(beta1_correction);
    float32x4_t beta2_corr_vec = vdupq_n_f32(beta2_correction);
    
    for (size_t i = 0; i < biases.size(); i += 4) {
        // Load vectors
        float32x4_t grad_vec = vld1q_f32(&grads[i]);
        float32x4_t m_vec = vld1q_f32(&m_t[i]);
        float32x4_t v_vec = vld1q_f32(&v_t[i]);
        
        // Update biased first moment
        float32x4_t m_new = vmlaq_f32(
            vmulq_f32(one_minus_beta1, grad_vec),
            beta1_vec, m_vec
        );
        
        // Update biased second moment
        float32x4_t grad_squared = vmulq_f32(grad_vec, grad_vec);
        float32x4_t v_new = vmlaq_f32(
            vmulq_f32(one_minus_beta2, grad_squared),
            beta2_vec, v_vec
        );
        
        // Compute bias-corrected moments
        float32x4_t m_hat = vdivq_f32(m_new, beta1_corr_vec);
        float32x4_t v_hat = vdivq_f32(v_new, beta2_corr_vec);
        
        // Compute update
        float32x4_t denom = vaddq_f32(vsqrtq_f32(v_hat), eps_vec);
        float32x4_t update = vdivq_f32(m_hat, denom);
        
        // Update biases
        float32x4_t bias_vec = vld1q_f32(&biases[i]);
        bias_vec = vsubq_f32(bias_vec, vmulq_f32(lr_vec, update));
        
        // Store results
        vst1q_f32(&biases[i], bias_vec);
        vst1q_f32(&m_t[i], m_new);
        vst1q_f32(&v_t[i], v_new);
    }
}

bool LSTMPredictor::are_adam_states_initialized() const {
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_num_classes = (num_classes + 3) & ~3;
    
    // Check FC layer Adam states
    if (m_fc_weight.empty() || v_fc_weight.empty() || 
        m_fc_bias.empty() || v_fc_bias.empty()) {
        return false;
    }

    // Check LSTM layer Adam states
    if (m_weight_ih.empty() || v_weight_ih.empty() ||
        m_weight_hh.empty() || v_weight_hh.empty() ||
        m_bias_ih.empty() || v_bias_ih.empty() ||
        m_bias_hh.empty() || v_bias_hh.empty()) {
        return false;
    }

    // Check aligned dimensions
    if (m_fc_weight.size() != aligned_num_classes || 
        m_fc_weight[0].size() != aligned_hidden_size) {
        return false;
    }

    // Check LSTM layer dimensions with alignment
    for (int layer = 0; layer < num_layers; ++layer) {
        const size_t aligned_input_size_layer = 
            (layer == 0) ? (input_size + 3) & ~3 : aligned_hidden_size;
        
        if (m_weight_ih[layer].size() != 4 * aligned_hidden_size ||
            m_weight_ih[layer][0].size() != aligned_input_size_layer ||
            m_weight_hh[layer].size() != 4 * aligned_hidden_size ||
            m_weight_hh[layer][0].size() != aligned_hidden_size ||
            m_bias_ih[layer].size() != 4 * aligned_hidden_size ||
            m_bias_hh[layer].size() != 4 * aligned_hidden_size) {
            return false;
        }
    }

    return adam_initialized;
}

void LSTMPredictor::set_weights(const std::vector<LSTMLayer>& weights) {
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_input_size = (input_size + 3) & ~3;
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        const size_t aligned_input_size_layer = 
            (layer == 0) ? aligned_input_size : aligned_hidden_size;
            
        // Resize with aligned dimensions
        lstm_layers[layer].weight_ih.resize(4 * aligned_hidden_size);
        lstm_layers[layer].weight_hh.resize(4 * aligned_hidden_size);
        
        for (auto& row : lstm_layers[layer].weight_ih) {
            row.resize(aligned_input_size_layer, 0.0f);
        }
        for (auto& row : lstm_layers[layer].weight_hh) {
            row.resize(aligned_hidden_size, 0.0f);
        }
        
        // Copy weights using NEON
        for (size_t i = 0; i < 4 * aligned_hidden_size; i += 4) {
            // Copy weight_ih
            for (size_t j = 0; j < aligned_input_size_layer; j += 4) {
                float32x4_t weight_vec = vld1q_f32(&weights[layer].weight_ih[i][j]);
                vst1q_f32(&lstm_layers[layer].weight_ih[i][j], weight_vec);
            }
            
            // Copy weight_hh
            for (size_t j = 0; j < aligned_hidden_size; j += 4) {
                float32x4_t weight_vec = vld1q_f32(&weights[layer].weight_hh[i][j]);
                vst1q_f32(&lstm_layers[layer].weight_hh[i][j], weight_vec);
            }
        }
        
        // Copy biases using NEON
        lstm_layers[layer].bias_ih.resize(4 * aligned_hidden_size, 0.0f);
        lstm_layers[layer].bias_hh.resize(4 * aligned_hidden_size, 0.0f);
        
        for (size_t i = 0; i < 4 * hidden_size; i += 4) {
            float32x4_t bias_ih_vec = vld1q_f32(&weights[layer].bias_ih[i]);
            float32x4_t bias_hh_vec = vld1q_f32(&weights[layer].bias_hh[i]);
            vst1q_f32(&lstm_layers[layer].bias_ih[i], bias_ih_vec);
            vst1q_f32(&lstm_layers[layer].bias_hh[i], bias_hh_vec);
        }
    }
}

void LSTMPredictor::train_step(const std::vector<std::vector<std::vector<float>>>& x,
                              const std::vector<float>& target,
                              float learning_rate) {
    try {
        // Input validation with aligned sizes
        const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
        const size_t aligned_input_size = (input_size + 3) & ~3;
        const size_t aligned_num_classes = (num_classes + 3) & ~3;
        
        // Dimension checks
        for (size_t batch = 0; batch < x.size(); ++batch) {
            for (size_t seq = 0; seq < x[batch].size(); ++seq) {
                if (x[batch][seq].size() != input_size) {
                    throw std::runtime_error(
                        "Input sequence dimension mismatch in train_step: batch " + 
                        std::to_string(batch) + ", seq " + std::to_string(seq));
                }
            }
        }
        
        // Initialize Adam if needed
        if (!are_adam_states_initialized()) {
            initialize_adam_states();
        }
        
        // Additional validation
        if (x.empty() || x[0].empty() || x[0][0].empty() ||
            x[0][0].size() != input_size || target.size() != num_classes) {
            throw std::runtime_error("Invalid input dimensions");
        }
        
        // Adam hyperparameters
        static int timestep = 0;
        timestep++;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-8f;

        // Forward pass with aligned memory
        auto lstm_output = forward(x);
        auto output = get_final_prediction(lstm_output);

        // Compute gradients
        auto grad_output = compute_mse_loss_gradient(output, target);
        const auto& last_hidden = lstm_output.final_hidden.back();

        // Backward pass through linear layer
        std::vector<std::vector<float>> fc_weight_grad;
        std::vector<float> fc_bias_grad;
        std::vector<float> lstm_grad;
        backward_linear_layer(grad_output, last_hidden, fc_weight_grad, fc_bias_grad, lstm_grad);

        // Verify dimensions and apply Adam updates
        if (fc_weight.size() != fc_weight_grad.size() || 
            fc_weight[0].size() != fc_weight_grad[0].size()) {
            throw std::runtime_error("FC layer dimension mismatch");
        }

        // Apply Adam updates using optimized functions
        apply_adam_update(fc_weight, fc_weight_grad, m_fc_weight, v_fc_weight,
                         learning_rate, beta1, beta2, epsilon, timestep);
        apply_adam_update(fc_bias, fc_bias_grad, m_fc_bias, v_fc_bias,
                         learning_rate, beta1, beta2, epsilon, timestep);

        // Validate cache and compute LSTM gradients
        if (layer_cache.empty() || lstm_grad.size() != hidden_size) {
            throw std::runtime_error("Invalid cache or gradient dimensions");
        }

        auto lstm_grads = backward_lstm_layer(lstm_grad, layer_cache, learning_rate);

        // Apply Adam updates to LSTM layers
        for (int layer = 0; layer < num_layers; ++layer) {
            if (lstm_layers[layer].weight_ih.size() != lstm_grads[layer].weight_ih_grad.size()) {
                throw std::runtime_error("LSTM dimension mismatch at layer " + 
                                       std::to_string(layer));
            }

            // Apply updates using NEON-optimized functions
            apply_adam_update(lstm_layers[layer].weight_ih, lstm_grads[layer].weight_ih_grad,
                            m_weight_ih[layer], v_weight_ih[layer],
                            learning_rate, beta1, beta2, epsilon, timestep);
            apply_adam_update(lstm_layers[layer].weight_hh, lstm_grads[layer].weight_hh_grad,
                            m_weight_hh[layer], v_weight_hh[layer],
                            learning_rate, beta1, beta2, epsilon, timestep);
            apply_adam_update(lstm_layers[layer].bias_ih, lstm_grads[layer].bias_ih_grad,
                            m_bias_ih[layer], v_bias_ih[layer],
                            learning_rate, beta1, beta2, epsilon, timestep);
            apply_adam_update(lstm_layers[layer].bias_hh, lstm_grads[layer].bias_hh_grad,
                            m_bias_hh[layer], v_bias_hh[layer],
                            learning_rate, beta1, beta2, epsilon, timestep);
        }

    } catch (const std::exception& e) {
        throw;
    }
}