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
    static float base_array[4], exp_array[4], result_array[4];
    vst1q_f32(base_array, base);
    vst1q_f32(exp_array, exp);
    
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
    static float a_array[4], b_array[4];
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
    
    // Pre-calculate aligned sizes
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_input_size = (input_size + 3) & ~3;
    const size_t aligned_num_classes = (num_classes + 3) & ~3;
    
    // Pre-allocate vectors with aligned sizes
    lstm_layers.resize(num_layers);
    last_gradients.resize(num_layers);
    h_state.resize(num_layers, std::vector<float>(aligned_hidden_size));
    c_state.resize(num_layers, std::vector<float>(aligned_hidden_size));
    
    // Pre-allocate FC layer vectors
    fc_weight.resize(aligned_num_classes, std::vector<float>(aligned_hidden_size));
    fc_bias.resize(aligned_num_classes);
    
    // Pre-allocate cache structure
    layer_cache.resize(num_layers);
    
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
    
    // Static vectors for reuse across all calls
    static std::vector<float> aligned_input;
    static std::vector<float> gates;
    static std::vector<float> output;
    static LSTMCacheEntry cache_entry;
    
    try {
        // Calculate aligned sizes
        const int expected_layer_input = (current_layer == 0) ? input_size : hidden_size;
        const size_t aligned_input_size = (expected_layer_input + 3) & ~3;
        const size_t aligned_hidden_size = (hidden_size + 3) & ~3;

        // Resize only if needed
        if (aligned_input.size() < aligned_input_size) {
            aligned_input.resize(aligned_input_size);
        }
        if (gates.size() < 4 * aligned_hidden_size) {
            gates.resize(4 * aligned_hidden_size);
        }
        if (output.size() < hidden_size) {
            output.resize(hidden_size);
        }
        
        // Zero out vectors
        std::fill(aligned_input.begin(), aligned_input.end(), 0.0f);
        std::fill(gates.begin(), gates.end(), 0.0f);

        // Create aligned input vector
        std::copy(input.begin(), input.end(), aligned_input.begin());

        // Verify dimensions
        if (layer.weight_ih.size() != 4 * aligned_hidden_size || 
            layer.weight_ih[0].size() != aligned_input_size) {
            throw std::runtime_error("Weight ih dimension mismatch");
        }

        if (layer.weight_hh.size() != 4 * aligned_hidden_size || 
            layer.weight_hh[0].size() != aligned_hidden_size) {
            throw std::runtime_error("Weight hh dimension mismatch");
        }

        // Verify state dimensions and ensure alignment
        if (h_state.size() != aligned_hidden_size) {
            h_state.resize(aligned_hidden_size, 0.0f);
        }
        if (c_state.size() != aligned_hidden_size) {
            c_state.resize(aligned_hidden_size, 0.0f);
        }

        // Set up cache entry if in training mode
        if (training_mode) {
            cache_entry.input = aligned_input;
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
        for (int h = 0; h < hidden_size; h += 4) {
            float32x4_t bias_ih_i = vld1q_f32(&layer.bias_ih[h]);
            float32x4_t bias_hh_i = vld1q_f32(&layer.bias_hh[h]);
            float32x4_t bias_ih_f = vld1q_f32(&layer.bias_ih[hidden_size + h]);
            float32x4_t bias_hh_f = vld1q_f32(&layer.bias_hh[hidden_size + h]);
            float32x4_t bias_ih_g = vld1q_f32(&layer.bias_ih[2 * hidden_size + h]);
            float32x4_t bias_hh_g = vld1q_f32(&layer.bias_hh[2 * hidden_size + h]);
            float32x4_t bias_ih_o = vld1q_f32(&layer.bias_ih[3 * hidden_size + h]);
            float32x4_t bias_hh_o = vld1q_f32(&layer.bias_hh[3 * hidden_size + h]);

            vst1q_f32(&gates[h], vaddq_f32(bias_ih_i, bias_hh_i));
            vst1q_f32(&gates[hidden_size + h], vaddq_f32(bias_ih_f, bias_hh_f));
            vst1q_f32(&gates[2 * hidden_size + h], vaddq_f32(bias_ih_g, bias_hh_g));
            vst1q_f32(&gates[3 * hidden_size + h], vaddq_f32(bias_ih_o, bias_hh_o));
        }

        // Process input-to-hidden contributions
        for (size_t i = 0; i < aligned_input_size; ++i) {
            float input_val = i < input.size() ? input[i] : 0.0f;
            float32x4_t input_val_vec = vdupq_n_f32(input_val);
            
            for (int h = 0; h < hidden_size; h += 4) {
                float32x4_t gates_i = vld1q_f32(&gates[h]);
                float32x4_t gates_f = vld1q_f32(&gates[hidden_size + h]);
                float32x4_t gates_g = vld1q_f32(&gates[2 * hidden_size + h]);
                float32x4_t gates_o = vld1q_f32(&gates[3 * hidden_size + h]);
                
                float32x4_t weight_i = vld1q_f32(&layer.weight_ih[h][i]);
                float32x4_t weight_f = vld1q_f32(&layer.weight_ih[hidden_size + h][i]);
                float32x4_t weight_g = vld1q_f32(&layer.weight_ih[2 * hidden_size + h][i]);
                float32x4_t weight_o = vld1q_f32(&layer.weight_ih[3 * hidden_size + h][i]);
                
                gates_i = vmlaq_f32(gates_i, weight_i, input_val_vec);
                gates_f = vmlaq_f32(gates_f, weight_f, input_val_vec);
                gates_g = vmlaq_f32(gates_g, weight_g, input_val_vec);
                gates_o = vmlaq_f32(gates_o, weight_o, input_val_vec);
                
                vst1q_f32(&gates[h], gates_i);
                vst1q_f32(&gates[hidden_size + h], gates_f);
                vst1q_f32(&gates[2 * hidden_size + h], gates_g);
                vst1q_f32(&gates[3 * hidden_size + h], gates_o);
            }
        }

        // Process hidden-to-hidden contributions
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

        // Create output vector
        std::memcpy(output.data(), h_state.data(), hidden_size * sizeof(float));

        // Store cache if in training mode
        if (training_mode) {
            layer_cache[current_layer][current_batch][current_timestep] = cache_entry;
        }

        return output;

    } catch (const std::exception& e) {
        throw;
    }
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
        
        // Reuse layer cache memory if possible
        if (training_mode) {
            if (layer_cache.size() != num_layers) {
                layer_cache.resize(num_layers);
            }
            
            for (int layer = 0; layer < num_layers; ++layer) {
                if (layer_cache[layer].size() != batch_size) {
                    layer_cache[layer].resize(batch_size);
                }
                for (size_t batch = 0; batch < batch_size; ++batch) {
                    if (layer_cache[layer][batch].size() != seq_len) {
                        layer_cache[layer][batch].resize(seq_len);
                    }
                }
            }
        }
        
        // Initialize output structure with preallocated memory
        LSTMOutput output;
        output.sequence_output.resize(batch_size);
        for (auto& batch_seq : output.sequence_output) {
            batch_seq.resize(seq_len);
            for (auto& seq_output : batch_seq) {
                seq_output.resize(hidden_size);
            }
        }
        
        // Reuse state vectors if possible
        if (!initial_hidden || !initial_cell) {
            if (h_state.size() != num_layers) {
                h_state.resize(num_layers);
                c_state.resize(num_layers);
            }
            
            for (int layer = 0; layer < num_layers; ++layer) {
                if (h_state[layer].size() != hidden_size) {
                    h_state[layer].resize(hidden_size, 0.0f);
                    c_state[layer].resize(hidden_size, 0.0f);
                } else {
                    std::fill(h_state[layer].begin(), h_state[layer].end(), 0.0f);
                    std::fill(c_state[layer].begin(), c_state[layer].end(), 0.0f);
                }
            }
        } else {
            h_state = *initial_hidden;
            c_state = *initial_cell;
        }
        
        // Static vector for layer input to avoid reallocations
        static std::vector<float> layer_input;
        
        // Process each batch and timestep
        for (size_t batch = 0; batch < batch_size; ++batch) {
            current_batch = batch;
            
            if (!initial_hidden || !initial_cell) {
                reset_states();
            }
            
            for (size_t t = 0; t < seq_len; ++t) {
                current_timestep = t;
                layer_input = x[batch][t];  // Reuse vector
                
                for (int layer = 0; layer < num_layers; ++layer) {
                    current_layer = layer;
                    layer_input = lstm_cell_forward(
                        layer_input,
                        h_state[layer],
                        c_state[layer],
                        lstm_layers[layer]
                    );
                }
                
                output.sequence_output[batch][t] = layer_input;
            }
        }
        
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
    std::vector<std::vector<float>>& fc_weight_grad,
    std::vector<float>& fc_bias_grad,
    std::vector<float>& lstm_grad) {
    
    static std::vector<float> aligned_grad_output;
    static std::vector<float> aligned_last_hidden;
    
    try {
        // Get aligned sizes
        const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
        const size_t aligned_num_classes = (num_classes + 3) & ~3;
        
        // Resize only if needed
        if (aligned_grad_output.size() < aligned_num_classes) {
            aligned_grad_output.resize(aligned_num_classes);
        }
        if (aligned_last_hidden.size() < aligned_hidden_size) {
            aligned_last_hidden.resize(aligned_hidden_size);
        }
        
        // Zero out vectors
        std::fill(aligned_grad_output.begin(), aligned_grad_output.end(), 0.0f);
        std::fill(aligned_last_hidden.begin(), aligned_last_hidden.end(), 0.0f);
        
        // Create aligned vectors
        std::copy(grad_output.begin(), grad_output.end(), aligned_grad_output.begin());
        std::copy(last_hidden.begin(), last_hidden.end(), aligned_last_hidden.begin());
        
        // Ensure fc_weight has correct dimensions
        if (fc_weight.size() != aligned_num_classes) {
            fc_weight.resize(aligned_num_classes, std::vector<float>(aligned_hidden_size, 0.0f));
        }
        for (auto& row : fc_weight) {
            if (row.size() != aligned_hidden_size) {
                row.resize(aligned_hidden_size, 0.0f);
            }
        }
        
        // Compute gradients using NEON
        for (size_t i = 0; i < aligned_num_classes; i += 4) {
            float32x4_t grad_vec = vld1q_f32(&aligned_grad_output[i]);
            
            // Copy bias gradients
            vst1q_f32(&fc_bias_grad[i], grad_vec);
            
            // Compute weight gradients
            for (size_t j = 0; j < aligned_hidden_size; j += 4) {
                float32x4_t hidden_vec = vld1q_f32(&aligned_last_hidden[j]);
                
                float grad_vals[4];
                vst1q_f32(grad_vals, grad_vec);
                
                for (size_t k = 0; k < 4 && i + k < aligned_num_classes; ++k) {
                    float32x4_t grad_val_vec = vdupq_n_f32(grad_vals[k]);
                    float32x4_t curr_grad = vmulq_f32(grad_val_vec, hidden_vec);
                    vst1q_f32(&fc_weight_grad[i + k][j], curr_grad);
                }
            }
        }
        
        // Compute LSTM gradients
        for (size_t i = 0; i < aligned_hidden_size; i += 4) {
            float32x4_t lstm_grad_vec = vdupq_n_f32(0.0f);
            
            for (size_t j = 0; j < num_classes; ++j) {
                float grad_val = aligned_grad_output[j];
                float32x4_t weight_vec = vld1q_f32(&fc_weight[j][i]);
                lstm_grad_vec = vmlaq_n_f32(lstm_grad_vec, weight_vec, grad_val);
            }
            
            vst1q_f32(&lstm_grad[i], lstm_grad_vec);
        }
        
    } catch (const std::exception& e) {
        throw;
    }
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
    
    // Initialize FC layer with aligned dimensions
    fc_weight.resize(num_classes, std::vector<float>(aligned_hidden_size, 0.0f));
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
        for (int j = 0; j < aligned_hidden_size; j += 4) {
            float32x4_t rand_vec = {dist(gen), dist(gen), dist(gen), dist(gen)};
            for (int k = 0; k < 4 && i + k < num_classes; ++k) {
                vst1q_f32(&fc_weight[i + k][j], rand_vec);
            }
        }
    }

    // Initialize LSTM layers with aligned dimensions
    lstm_layers.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        const size_t aligned_input_size_layer = 
            (layer == 0) ? aligned_input_size : aligned_hidden_size;
        
        // Initialize with aligned dimensions
        lstm_layers[layer].weight_ih.resize(4 * aligned_hidden_size, 
            std::vector<float>(aligned_input_size_layer, 0.0f));
        lstm_layers[layer].weight_hh.resize(4 * aligned_hidden_size, 
            std::vector<float>(aligned_hidden_size, 0.0f));
        lstm_layers[layer].bias_ih.resize(4 * aligned_hidden_size, 0.0f);
        lstm_layers[layer].bias_hh.resize(4 * aligned_hidden_size, 0.0f);
        
        // Initialize weights and biases using NEON
        for (int i = 0; i < 4 * aligned_hidden_size; i += 4) {
            // Initialize biases
            float32x4_t rand_bias_ih = {dist(gen), dist(gen), dist(gen), dist(gen)};
            float32x4_t rand_bias_hh = {dist(gen), dist(gen), dist(gen), dist(gen)};
            vst1q_f32(&lstm_layers[layer].bias_ih[i], rand_bias_ih);
            vst1q_f32(&lstm_layers[layer].bias_hh[i], rand_bias_hh);
            
            // Initialize input-hidden weights
            for (int j = 0; j < aligned_input_size_layer; j += 4) {
                float32x4_t rand_vec = {dist(gen), dist(gen), dist(gen), dist(gen)};
                vst1q_f32(&lstm_layers[layer].weight_ih[i][j], rand_vec);
            }
            
            // Initialize hidden-hidden weights
            for (int j = 0; j < aligned_hidden_size; j += 4) {
                float32x4_t rand_vec = {dist(gen), dist(gen), dist(gen), dist(gen)};
                vst1q_f32(&lstm_layers[layer].weight_hh[i][j], rand_vec);
            }
        }
    }
}

void LSTMPredictor::initialize_adam_states() {
    const size_t aligned_hidden_size = (hidden_size + 3) & ~3;
    const size_t aligned_input_size = (input_size + 3) & ~3;
    const size_t aligned_num_classes = (num_classes + 3) & ~3;
    
    try {
        // Static allocation for temporary vectors
        static std::vector<float> temp_vector;
        const size_t max_vector_size = std::max({
            aligned_hidden_size * 4,
            aligned_input_size,
            aligned_num_classes
        });
        
        if (temp_vector.size() < max_vector_size) {
            temp_vector.resize(max_vector_size, 0.0f);
        }
        
        // Initialize FC layer vectors with aligned sizes
        m_fc_weight.resize(aligned_num_classes);
        v_fc_weight.resize(aligned_num_classes);
        
        // Use NEON to zero out vectors in blocks
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (size_t i = 0; i < max_vector_size; i += 4) {
            vst1q_f32(&temp_vector[i], zero_vec);
        }
        
        // Use the zeroed temp vector to initialize matrices
        for (auto& row : m_fc_weight) {
            row.assign(temp_vector.begin(), temp_vector.begin() + aligned_hidden_size);
        }
        for (auto& row : v_fc_weight) {
            row.assign(temp_vector.begin(), temp_vector.begin() + aligned_hidden_size);
        }
        
        // Initialize bias vectors
        m_fc_bias.assign(temp_vector.begin(), temp_vector.begin() + aligned_num_classes);
        v_fc_bias.assign(temp_vector.begin(), temp_vector.begin() + aligned_num_classes);
        
        // Initialize LSTM layer states
        m_weight_ih.resize(num_layers);
        v_weight_ih.resize(num_layers);
        m_weight_hh.resize(num_layers);
        v_weight_hh.resize(num_layers);
        m_bias_ih.resize(num_layers);
        v_bias_ih.resize(num_layers);
        m_bias_hh.resize(num_layers);
        v_bias_hh.resize(num_layers);
        
        for (int layer = 0; layer < num_layers; ++layer) {
            const size_t aligned_input_size_layer = (layer == 0) ? aligned_input_size : aligned_hidden_size;
            
            // Initialize matrices using temp_vector
            for (auto* matrices : {&m_weight_ih[layer], &v_weight_ih[layer],
                                 &m_weight_hh[layer], &v_weight_hh[layer]}) {
                matrices->resize(4 * aligned_hidden_size);
                for (auto& row : *matrices) {
                    row.assign(temp_vector.begin(), 
                             temp_vector.begin() + 
                             (matrices == &m_weight_ih[layer] || matrices == &v_weight_ih[layer] ? 
                              aligned_input_size_layer : aligned_hidden_size));
                }
            }
            
            // Initialize bias vectors
            for (auto* biases : {&m_bias_ih[layer], &v_bias_ih[layer],
                                &m_bias_hh[layer], &v_bias_hh[layer]}) {
                biases->assign(temp_vector.begin(), temp_vector.begin() + 4 * aligned_hidden_size);
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
    
    // Static NEON vectors to avoid reallocation
    static float32x4_t beta1_vec, one_minus_beta1, beta2_vec, one_minus_beta2;
    static float32x4_t eps_vec, lr_vec, beta1_corr_vec, beta2_corr_vec;
    
    // Update constants only when t changes
    static int last_t = 0;
    if (t != last_t) {
        beta1_vec = vdupq_n_f32(beta1);
        one_minus_beta1 = vdupq_n_f32(1.0f - beta1);
        beta2_vec = vdupq_n_f32(beta2);
        one_minus_beta2 = vdupq_n_f32(1.0f - beta2);
        eps_vec = vdupq_n_f32(epsilon);
        lr_vec = vdupq_n_f32(learning_rate);
        
        float beta1_correction = 1.0f - pow_float(beta1, static_cast<float>(t));
        float beta2_correction = 1.0f - pow_float(beta2, static_cast<float>(t));
        beta1_corr_vec = vdupq_n_f32(beta1_correction);
        beta2_corr_vec = vdupq_n_f32(beta2_correction);
        
        last_t = t;
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); j += 4) {
            // Process 4 elements at once using NEON
            float32x4_t grad_vec = vld1q_f32(&grads[i][j]);
            float32x4_t m_vec = vld1q_f32(&m_t[i][j]);
            float32x4_t v_vec = vld1q_f32(&v_t[i][j]);
            float32x4_t weight_vec = vld1q_f32(&weights[i][j]);
            
            // Fused multiply-add operations
            float32x4_t m_new = vmlaq_f32(vmulq_f32(one_minus_beta1, grad_vec), beta1_vec, m_vec);
            float32x4_t v_new = vmlaq_f32(vmulq_f32(one_minus_beta2, vmulq_f32(grad_vec, grad_vec)), beta2_vec, v_vec);
            
            // Compute update
            float32x4_t m_hat = vdivq_f32(m_new, beta1_corr_vec);
            float32x4_t v_hat = vdivq_f32(v_new, beta2_corr_vec);
            float32x4_t update = vdivq_f32(m_hat, vaddq_f32(vsqrtq_f32(v_hat), eps_vec));
            
            // Apply update
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
    
    // Static NEON vectors to avoid reallocation
    static float32x4_t beta1_vec, one_minus_beta1, beta2_vec, one_minus_beta2;
    static float32x4_t eps_vec, lr_vec, beta1_corr_vec, beta2_corr_vec;
    
    // Update constants only when t changes
    static int last_t = 0;
    if (t != last_t) {
        beta1_vec = vdupq_n_f32(beta1);
        one_minus_beta1 = vdupq_n_f32(1.0f - beta1);
        beta2_vec = vdupq_n_f32(beta2);
        one_minus_beta2 = vdupq_n_f32(1.0f - beta2);
        eps_vec = vdupq_n_f32(epsilon);
        lr_vec = vdupq_n_f32(learning_rate);
        
        float beta1_correction = 1.0f - pow_float(beta1, static_cast<float>(t));
        float beta2_correction = 1.0f - pow_float(beta2, static_cast<float>(t));
        beta1_corr_vec = vdupq_n_f32(beta1_correction);
        beta2_corr_vec = vdupq_n_f32(beta2_correction);
        
        last_t = t;
    }
    
    // Process bias updates in parallel blocks
    #pragma omp parallel for
    for (size_t i = 0; i < biases.size(); i += 16) {
        // Process 4 elements at a time using NEON, with prefetching
        for (size_t j = i; j < std::min(i + 16, biases.size()); j += 4) {
            // Prefetch next iteration's data
            if (j + 4 < biases.size()) {
                __builtin_prefetch(&grads[j + 4], 0, 3);
                __builtin_prefetch(&m_t[j + 4], 1, 3);
                __builtin_prefetch(&v_t[j + 4], 1, 3);
                __builtin_prefetch(&biases[j + 4], 1, 3);
            }
            
            // Load vectors
            float32x4_t grad_vec = vld1q_f32(&grads[j]);
            float32x4_t m_vec = vld1q_f32(&m_t[j]);
            float32x4_t v_vec = vld1q_f32(&v_t[j]);
            float32x4_t bias_vec = vld1q_f32(&biases[j]);
            
            // Fused multiply-add operations for moment updates
            float32x4_t m_new = vmlaq_f32(
                vmulq_f32(one_minus_beta1, grad_vec),
                beta1_vec, m_vec
            );
            
            float32x4_t grad_squared = vmulq_f32(grad_vec, grad_vec);
            float32x4_t v_new = vmlaq_f32(
                vmulq_f32(one_minus_beta2, grad_squared),
                beta2_vec, v_vec
            );
            
            // Compute bias-corrected moments and update in one go
            float32x4_t m_hat = vdivq_f32(m_new, beta1_corr_vec);
            float32x4_t v_hat = vdivq_f32(v_new, beta2_corr_vec);
            float32x4_t denom = vaddq_f32(vsqrtq_f32(v_hat), eps_vec);
            float32x4_t update = vdivq_f32(m_hat, denom);
            
            // Apply update to biases
            bias_vec = vsubq_f32(bias_vec, vmulq_f32(lr_vec, update));
            
            // Store results
            vst1q_f32(&biases[j], bias_vec);
            vst1q_f32(&m_t[j], m_new);
            vst1q_f32(&v_t[j], v_new);
        }
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
                    throw std::runtime_error("Input sequence dimension mismatch");
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
        auto prediction = get_final_prediction(lstm_output);
        
        // Compute gradients
        std::vector<float> grad_output(num_classes);
        for (size_t i = 0; i < num_classes; ++i) {
            grad_output[i] = prediction[i] - target[i];
        }

        // Run backward pass
        std::vector<std::vector<float>> fc_weight_grad;
        std::vector<float> fc_bias_grad;
        std::vector<float> lstm_grad;
        
        backward_linear_layer(grad_output, lstm_output.sequence_output.back().back(),
                            fc_weight_grad, fc_bias_grad, lstm_grad);

        // Apply Adam updates
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
                throw std::runtime_error("LSTM dimension mismatch");
            }

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

LSTMPredictor::~LSTMPredictor() {
    try {
        // Add debug output
        std::cout << "Starting LSTMPredictor cleanup..." << std::endl;
        
        // Clear vectors safely with size checks
        if (!lstm_layers.empty()) {
            std::cout << "Clearing lstm_layers..." << std::endl;
            lstm_layers.clear();
        }
        
        if (!last_gradients.empty()) {
            std::cout << "Clearing last_gradients..." << std::endl;
            last_gradients.clear();
        }
        
        if (!h_state.empty()) {
            std::cout << "Clearing h_state..." << std::endl;
            h_state.clear();
        }
        
        if (!c_state.empty()) {
            std::cout << "Clearing c_state..." << std::endl;
            c_state.clear();
        }
        
        if (!layer_cache.empty()) {
            std::cout << "Clearing layer_cache..." << std::endl;
            layer_cache.clear();
        }
        
        if (!fc_weight.empty()) {
            std::cout << "Clearing fc_weight..." << std::endl;
            fc_weight.clear();
        }
        
        if (!fc_bias.empty()) {
            std::cout << "Clearing fc_bias..." << std::endl;
            fc_bias.clear();
        }
        
        // Clear Adam states if they exist
        if (!m_fc_weight.empty()) {
            std::cout << "Clearing Adam states..." << std::endl;
            m_fc_weight.clear();
            v_fc_weight.clear();
            m_fc_bias.clear();
            v_fc_bias.clear();
            m_weight_ih.clear();
            v_weight_ih.clear();
            m_weight_hh.clear();
            v_weight_hh.clear();
            m_bias_ih.clear();
            v_bias_ih.clear();
            m_bias_hh.clear();
            v_bias_hh.clear();
        }
        
        std::cout << "LSTMPredictor cleanup completed." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during LSTMPredictor cleanup: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error during LSTMPredictor cleanup" << std::endl;
    }
}