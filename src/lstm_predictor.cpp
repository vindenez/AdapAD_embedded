#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "config.hpp"
#include "model_state.hpp"

#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <execinfo.h>
#include <cxxabi.h> 

inline float pow_float(float base, float exp) {
    return std::pow(base, exp);
}

LSTMPredictor::LSTMPredictor(int num_classes, int input_size, int hidden_size, 
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
        velocity_weight_ih[layer].resize(4 * hidden_size, std::vector<float>(input_size_layer, 0.0f));
        velocity_weight_hh[layer].resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        velocity_bias_ih[layer].resize(4 * hidden_size, 0.0f);
        velocity_bias_hh[layer].resize(4 * hidden_size, 0.0f);
    }
    
    velocity_fc_weight.resize(num_classes, std::vector<float>(hidden_size, 0.0f));
    velocity_fc_bias.resize(num_classes, 0.0f);
    
    // Pre-allocate layer cache with minimal structure
    layer_cache.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        layer_cache[layer].resize(1);  // One batch
        layer_cache[layer][0].resize(lookback_len);  // Sequence length
        
        // Pre-allocate each cache entry with properly sized vectors
        for (auto& seq : layer_cache[layer][0]) {
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
    for (auto& grad : last_gradients) {
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

void LSTMPredictor::reset_states() {
    c_state.clear();
    h_state.clear();
    c_state.resize(num_layers, std::vector<float>(hidden_size, 0.0f));
    h_state.resize(num_layers, std::vector<float>(hidden_size, 0.0f));
}

float LSTMPredictor::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float LSTMPredictor::tanh_custom(float x) {
    return std::tanh(x);
}

std::vector<float> LSTMPredictor::lstm_cell_forward(
    const std::vector<float>& input,
    std::vector<float>& h_state,
    std::vector<float>& c_state,
    const LSTMLayer& layer) {

    // Get the correct input size for this layer
    int expected_layer_input = (current_layer == 0) ? input_size : hidden_size;

    // Verify weight dimensions
    int weight_input_size = layer.weight_ih[0].size();
    if (weight_input_size != expected_layer_input) {
        throw std::runtime_error("Weight dimension mismatch in lstm_cell_forward");
    }

    // Verify input size
    if (input.size() != expected_layer_input) {
        throw std::runtime_error("Input size mismatch in lstm_cell_forward");
    }

    // Verify state dimensions
    if (h_state.size() != hidden_size) {
        h_state.resize(hidden_size, 0.0f);
    }
    if (c_state.size() != hidden_size) {
        c_state.resize(hidden_size, 0.0f);
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
    
    // Reference to cache entry if in training or online learning mode
    LSTMCacheEntry* cache_entry = nullptr;
    if (training_mode) {  // This includes both training and online learning
        // Validate indices before accessing cache
        if (current_layer >= layer_cache.size() ||
            current_batch >= layer_cache[current_layer].size() ||
            current_timestep >= layer_cache[current_layer][current_batch].size()) {
            throw std::runtime_error("Invalid cache access");
        }
        
        cache_entry = &layer_cache[current_layer][current_batch][current_timestep];
        
        // Resize vectors to exact needed size
        cache_entry->input.resize(expected_layer_input);
        cache_entry->prev_hidden.resize(hidden_size);
        cache_entry->prev_cell.resize(hidden_size);
        cache_entry->cell_state.resize(hidden_size);
        cache_entry->input_gate.resize(hidden_size);
        cache_entry->forget_gate.resize(hidden_size);
        cache_entry->cell_candidate.resize(hidden_size);
        cache_entry->output_gate.resize(hidden_size);
        cache_entry->hidden_state.resize(hidden_size);
        
        // Copy input and states
        std::copy(input.begin(), input.end(), cache_entry->input.begin());
        std::copy(h_state.begin(), h_state.end(), cache_entry->prev_hidden.begin());
        std::copy(c_state.begin(), c_state.end(), cache_entry->prev_cell.begin());
    }
    
    // Initialize gates with biases (PyTorch layout: [i,f,g,o])
    std::vector<float> gates(4 * hidden_size);
    for (int h = 0; h < hidden_size; ++h) {
        gates[h] = layer.bias_ih[h] + layer.bias_hh[h];                                         // input gate (i)
        gates[hidden_size + h] = layer.bias_ih[hidden_size + h] + 
                                layer.bias_hh[hidden_size + h];                                 // forget gate (f)
        gates[2 * hidden_size + h] = layer.bias_ih[2 * hidden_size + h] + 
                                    layer.bias_hh[2 * hidden_size + h];                         // cell candidate (Ĉ)
        gates[3 * hidden_size + h] = layer.bias_ih[3 * hidden_size + h] + 
                                    layer.bias_hh[3 * hidden_size + h];                         // output gate (o)
    }
    
    // Input to hidden contributions
    for (size_t i = 0; i < input.size(); ++i) {
        for (int h = 0; h < hidden_size; ++h) {
            gates[h] += layer.weight_ih[h][i] * input[i];                                       // input gate
            gates[hidden_size + h] += layer.weight_ih[hidden_size + h][i] * input[i];           // forget gate
            gates[2 * hidden_size + h] += layer.weight_ih[2 * hidden_size + h][i] * input[i];   // cell candidate
            gates[3 * hidden_size + h] += layer.weight_ih[3 * hidden_size + h][i] * input[i];   // output gate
        }
    }
    
    // Hidden to hidden contributions
    for (int h = 0; h < hidden_size; ++h) {
        for (size_t i = 0; i < hidden_size; ++i) {
            gates[h] += layer.weight_hh[h][i] * h_state[i];                                     // input gate
            gates[hidden_size + h] += layer.weight_hh[hidden_size + h][i] * h_state[i];         // forget gate
            gates[2 * hidden_size + h] += layer.weight_hh[2 * hidden_size + h][i] * h_state[i]; // cell candidate
            gates[3 * hidden_size + h] += layer.weight_hh[3 * hidden_size + h][i] * h_state[i]; // output gate
        }
    }

    // Apply activations and update states
    for (int h = 0; h < hidden_size; ++h) {
        float i_t = sigmoid(gates[h]);                                                          // input gate
        float f_t = sigmoid(gates[hidden_size + h]);                                            // forget gate
        float cell_candidate = tanh_custom(gates[2 * hidden_size + h]);                         // cell candidate
        float o_t = sigmoid(gates[3 * hidden_size + h]);                                        // output gate

        // Update cell state
        float new_cell = f_t * c_state[h] + i_t * cell_candidate;
        c_state[h] = new_cell;
        
        // Update hidden state
        float new_hidden = o_t * tanh_custom(new_cell);
        h_state[h] = new_hidden;

        // Store values in cache if in training mode (includes online learning)
        if (training_mode && cache_entry) {
            cache_entry->input_gate[h] = i_t;
            cache_entry->forget_gate[h] = f_t;
            cache_entry->cell_candidate[h] = cell_candidate;
            cache_entry->output_gate[h] = o_t;
            cache_entry->cell_state[h] = new_cell;
            cache_entry->hidden_state[h] = new_hidden;
        }
    }
    
    // Ensure h_state has the correct size
    if (h_state.size() != hidden_size) {
        h_state.resize(hidden_size);
    }

    // Create a copy of h_state to return
    std::vector<float> output = h_state;
    
    // Verify output dimensions one last time
    if (output.size() != hidden_size) {
        throw std::runtime_error("Output size mismatch in lstm_cell_forward");
    }

    return output;
}


LSTMPredictor::LSTMOutput LSTMPredictor::forward(
    const std::vector<std::vector<std::vector<float>>>& x,
    const std::vector<std::vector<float>>* initial_hidden,
    const std::vector<std::vector<float>>* initial_cell) {

    for (size_t batch = 0; batch < x.size(); ++batch) {
        for (size_t seq = 0; seq < x[batch].size(); ++seq) {
            // Verify input dimensions for each timestep
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
        
        // Initialize output structure with minimal size
        LSTMOutput output;
        output.sequence_output.resize(batch_size);
        for (auto& batch : output.sequence_output) {
            batch.resize(seq_len);
            for (auto& seq : batch) {
                seq.resize(hidden_size, 0.0f);
            }
        }
        
        // Process each batch
        for (size_t batch = 0; batch < batch_size; ++batch) {
            current_batch = batch;
            
            if (!initial_hidden || !initial_cell) {
                reset_states();
            }
            
            for (size_t t = 0; t < seq_len; ++t) {
                current_timestep = t;
                
                std::vector<float> layer_input = x[batch][t];
                std::vector<std::vector<float>> layer_outputs(num_layers + 1);
                layer_outputs[0] = layer_input;
                
                // Process through LSTM layers
                for (int layer = 0; layer < num_layers; ++layer) {
                    current_layer = layer;
                    
                    // Get correct input size for this layer
                    int expected_input_size = (layer == 0) ? input_size : hidden_size;
                    
                    // Verify dimensions before forward pass
                    if (layer_outputs[layer].size() != expected_input_size) {
                        throw std::runtime_error("Layer input dimension mismatch at layer " + 
                                               std::to_string(static_cast<long long>(layer)));
                    }
                    
                    // Ensure cache entry exists and is properly sized
                    if (training_mode) {
                        // Validate indices before accessing cache
                        if (current_layer >= layer_cache.size() ||
                            current_batch >= layer_cache[current_layer].size() ||
                            current_timestep >= layer_cache[current_layer][current_batch].size()) {
                            throw std::runtime_error("Invalid cache access");
                        }
                        
                        auto& cache_entry = layer_cache[current_layer][current_batch][current_timestep];
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
                    
                    layer_outputs[layer + 1] = lstm_cell_forward(
                        layer_outputs[layer],
                        h_state[layer],
                        c_state[layer],
                        lstm_layers[layer]
                    );
                }
                
                output.sequence_output[batch][t] = layer_outputs[num_layers];
            }
        }
        
        output.final_hidden = h_state;
        output.final_cell = c_state;
        
        return output;
        
    } catch (const std::exception& e) {
        // Clean up on error
        clear_training_state();
        throw;
    }
}

// Setter methods for loading trained weights
void LSTMPredictor::set_lstm_weights(int layer, 
                                   const std::vector<std::vector<float>>& w_ih,
                                   const std::vector<std::vector<float>>& w_hh) {
    if (layer < num_layers) {
        lstm_layers[layer].weight_ih = w_ih;
        lstm_layers[layer].weight_hh = w_hh;
    }
}

void LSTMPredictor::set_lstm_bias(int layer,
                                 const std::vector<float>& b_ih,
                                 const std::vector<float>& b_hh) {
    if (layer < num_layers) {
        lstm_layers[layer].bias_ih = b_ih;
        lstm_layers[layer].bias_hh = b_hh;
    }
}

void LSTMPredictor::set_fc_weights(const std::vector<std::vector<float>>& weights,
                                  const std::vector<float>& bias) {
    fc_weight = weights;
    fc_bias = bias;
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
    
    // Initialize gradients with correct dimensions
    weight_grad.resize(num_classes, std::vector<float>(hidden_size, 0.0f));
    bias_grad = grad_output;  // Copy gradient directly for bias
    input_grad.resize(hidden_size, 0.0f);
    
    // Compute weight gradients
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            weight_grad[i][j] = grad_output[i] * last_hidden[j];
        }
    }
    
    // Compute input gradients
    // Multiply fc_weight^T (4xnum_classes) with grad_output (num_classes)
    for (int i = 0; i < hidden_size; ++i) {
        input_grad[i] = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            input_grad[i] += fc_weight[j][i] * grad_output[j];
        }
    }
}

std::vector<LSTMPredictor::LSTMGradients> LSTMPredictor::backward_lstm_layer(
    const std::vector<float>& grad_output,
    const std::vector<std::vector<std::vector<LSTMCacheEntry>>>& cache,
    float learning_rate) {
    
    // Add dimension validation
    if (grad_output.size() != hidden_size) {
        throw std::runtime_error("grad_output size mismatch in backward_lstm_layer");
    }
    
    // Add cache validation
    if (cache.size() != num_layers) {
        throw std::runtime_error("cache layer count mismatch in backward_lstm_layer");
    }
    
    std::vector<LSTMGradients> layer_grads(num_layers);
    
    // Initialize gradients for each layer
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        layer_grads[layer].weight_ih_grad.resize(4 * hidden_size, std::vector<float>(input_size_layer, 0.0f));
        layer_grads[layer].weight_hh_grad.resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        layer_grads[layer].bias_ih_grad.resize(4 * hidden_size, 0.0f);
        layer_grads[layer].bias_hh_grad.resize(4 * hidden_size, 0.0f);

    }
    
    // Initialize dh_next and dc_next for each layer
    std::vector<std::vector<float>> dh_next(num_layers, std::vector<float>(hidden_size, 0.0f));
    std::vector<std::vector<float>> dc_next(num_layers, std::vector<float>(hidden_size, 0.0f));
    
    // Start from the last layer and move backward
    for (int layer = num_layers - 1; layer >= 0; --layer) {

        std::vector<float> dh = dh_next[layer];
        std::vector<float> dc = dc_next[layer];
        
        // Add bounds checking before accessing cache
        if (current_batch >= cache[layer].size()) {
            throw std::runtime_error("Cache batch index out of bounds");
        }
        
        const auto& layer_cache = cache[layer][current_batch];
        
        // If this is the last layer, add grad output (like dy @ Wy.T in PyTorch)
        if (layer == num_layers - 1) {
            for (int h = 0; h < hidden_size; ++h) {
                dh[h] += grad_output[h];
            }
        }

        // Process each time step in reverse order
        for (int t = layer_cache.size() - 1; t >= 0; --t) {

            const auto& cache_entry = layer_cache[t];
            
            std::vector<float> dh_prev(hidden_size, 0.0f);
            std::vector<float> dc_prev(hidden_size, 0.0f);

            // Process each hidden unit
            for (int h = 0; h < hidden_size; ++h) {
                float tanh_c = tanh_custom(cache_entry.cell_state[h]);
                float dho = dh[h];
                
                // 1. Cell state gradient
                float dc_t = dho * cache_entry.output_gate[h] * (1.0f - tanh_c * tanh_c);
                dc_t += dc[h];  // Add gradient from future timestep
                
                // 2. Gate gradients
                float do_t = dho * tanh_c * cache_entry.output_gate[h] * (1.0f - cache_entry.output_gate[h]);
                float di_t = dc_t * cache_entry.cell_candidate[h] * cache_entry.input_gate[h] * (1.0f - cache_entry.input_gate[h]);
                float df_t = dc_t * cache_entry.prev_cell[h] * cache_entry.forget_gate[h] * (1.0f - cache_entry.forget_gate[h]);
                float dcell_candidate = dc_t * cache_entry.input_gate[h] * (1.0f - cache_entry.cell_candidate[h] * cache_entry.cell_candidate[h]);
                

                // 3. Accumulate weight gradients
                int input_size_layer = (layer == 0) ? input_size : hidden_size;
                for (int j = 0; j < input_size_layer; ++j) {
                    float input_j = cache_entry.input[j];
                    layer_grads[layer].weight_ih_grad[h][j] += di_t * input_j;
                    layer_grads[layer].weight_ih_grad[hidden_size + h][j] += df_t * input_j;
                    layer_grads[layer].weight_ih_grad[2 * hidden_size + h][j] += dcell_candidate * input_j;
                    layer_grads[layer].weight_ih_grad[3 * hidden_size + h][j] += do_t * input_j;
                }
                
                // 4. Accumulate hidden-hidden weight gradients
                for (int j = 0; j < hidden_size; ++j) {
                    float h_prev_j = cache_entry.prev_hidden[j];
                    layer_grads[layer].weight_hh_grad[h][j] += di_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[hidden_size + h][j] += df_t * h_prev_j;
                    layer_grads[layer].weight_hh_grad[2 * hidden_size + h][j] += dcell_candidate * h_prev_j;
                    layer_grads[layer].weight_hh_grad[3 * hidden_size + h][j] += do_t * h_prev_j;
                    
                    // Accumulate gradients for next timestep's hidden state
                    dh_prev[j] += di_t * lstm_layers[layer].weight_hh[h][j];
                    dh_prev[j] += df_t * lstm_layers[layer].weight_hh[hidden_size + h][j];
                    dh_prev[j] += dcell_candidate * lstm_layers[layer].weight_hh[2 * hidden_size + h][j];
                    dh_prev[j] += do_t * lstm_layers[layer].weight_hh[3 * hidden_size + h][j];
                }
                
                // 5. Accumulate bias gradients
                layer_grads[layer].bias_ih_grad[h] += di_t;
                layer_grads[layer].bias_ih_grad[hidden_size + h] += df_t;
                layer_grads[layer].bias_ih_grad[2 * hidden_size + h] += dcell_candidate;
                layer_grads[layer].bias_ih_grad[3 * hidden_size + h] += do_t;
                
                // 6. Cell state gradient for previous timestep
                dc_prev[h] = dc_t * cache_entry.forget_gate[h];
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


void LSTMPredictor::train_step(const std::vector<std::vector<std::vector<float>>>& x,
                              const std::vector<float>& target,
                              const LSTMOutput& lstm_output,
                              float learning_rate) {
    try {
        // Add detailed dimension checking for each sequence step
        for (size_t batch = 0; batch < x.size(); ++batch) {
            for (size_t seq = 0; seq < x[batch].size(); ++seq) {
                if (x[batch][seq].size() != input_size) {
                    throw std::runtime_error(
                        "Input sequence dimension mismatch in train_step: batch " + 
                        std::to_string(batch) + ", seq " + std::to_string(seq));
                }
            }
        }
        
        // Verify input dimensions
        if (x.empty() || x[0].empty() || x[0][0].empty()) {
            throw std::runtime_error("Empty input tensor");
        }
        if (x[0][0].size() != input_size) {
            throw std::runtime_error("Input feature size mismatch");
        }
        
        // Verify target dimensions
        if (target.size() != num_classes) {
            throw std::invalid_argument("Target size mismatch");
        }
        
        // Get final prediction (no forward pass needed)
        auto output = get_final_prediction(lstm_output);

        // Compute gradients
        auto grad_output = compute_mse_loss_gradient(output, target);

        // Extract final hidden state
        const auto& last_hidden = lstm_output.sequence_output.back().back();

        // Backward pass through linear layer
        std::vector<std::vector<float>> fc_weight_grad;
        std::vector<float> fc_bias_grad;
        std::vector<float> lstm_grad;
        backward_linear_layer(grad_output, last_hidden, fc_weight_grad, fc_bias_grad, lstm_grad);

        // Update FC layer weights using SGD with momentum
        apply_sgd_update(fc_weight, fc_weight_grad, learning_rate, 0.9f);
        apply_sgd_update(fc_bias, fc_bias_grad, learning_rate, 0.9f);

        // Get LSTM layer gradients
        auto lstm_gradients = backward_lstm_layer(lstm_grad, layer_cache, learning_rate);

        // Update LSTM layer parameters using SGD with momentum
        for (int layer = 0; layer < num_layers; ++layer) {
            if (layer >= lstm_layers.size()) {
                throw std::runtime_error("Layer index out of bounds: " + std::to_string(layer));
            }
            if (layer >= lstm_gradients.size()) {
                throw std::runtime_error("Gradient index out of bounds: " + std::to_string(layer));
            }
            
            // Verify dimensions before updates
            if (lstm_layers[layer].weight_ih.size() != lstm_gradients[layer].weight_ih_grad.size() ||
                lstm_layers[layer].weight_hh.size() != lstm_gradients[layer].weight_hh_grad.size() ||
                lstm_layers[layer].bias_ih.size() != lstm_gradients[layer].bias_ih_grad.size() ||
                lstm_layers[layer].bias_hh.size() != lstm_gradients[layer].bias_hh_grad.size()) {
                throw std::runtime_error("Gradient dimension mismatch in layer " + std::to_string(layer));
            }

            apply_sgd_update(lstm_layers[layer].weight_ih, lstm_gradients[layer].weight_ih_grad, learning_rate, 0.9f);
            apply_sgd_update(lstm_layers[layer].weight_hh, lstm_gradients[layer].weight_hh_grad, learning_rate, 0.9f);
            apply_sgd_update(lstm_layers[layer].bias_ih, lstm_gradients[layer].bias_ih_grad, learning_rate, 0.9f);
            apply_sgd_update(lstm_layers[layer].bias_hh, lstm_gradients[layer].bias_hh_grad, learning_rate, 0.9f);
        }

        // Only clear temporary data, not the entire state
        if (online_learning_mode) {
            // Clear gradients but maintain structure
            for (auto& grad : last_gradients) {
                for (auto& row : grad.weight_ih_grad) {
                    std::fill(row.begin(), row.end(), 0.0f);
                }
                for (auto& row : grad.weight_hh_grad) {
                    std::fill(row.begin(), row.end(), 0.0f);
                }
                std::fill(grad.bias_ih_grad.begin(), grad.bias_ih_grad.end(), 0.0f);
                std::fill(grad.bias_hh_grad.begin(), grad.bias_hh_grad.end(), 0.0f);
            }
            
            // Clear layer cache but maintain structure
            for (auto& layer : layer_cache) {
                for (auto& batch : layer) {
                    for (auto& seq : batch) {
                        std::fill(seq.input.begin(), seq.input.end(), 0.0f);
                        std::fill(seq.prev_hidden.begin(), seq.prev_hidden.end(), 0.0f);
                        std::fill(seq.prev_cell.begin(), seq.prev_cell.end(), 0.0f);
                        std::fill(seq.cell_state.begin(), seq.cell_state.end(), 0.0f);
                        std::fill(seq.input_gate.begin(), seq.input_gate.end(), 0.0f);
                        std::fill(seq.forget_gate.begin(), seq.forget_gate.end(), 0.0f);
                        std::fill(seq.cell_candidate.begin(), seq.cell_candidate.end(), 0.0f);
                        std::fill(seq.output_gate.begin(), seq.output_gate.end(), 0.0f);
                        std::fill(seq.hidden_state.begin(), seq.hidden_state.end(), 0.0f);
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        // Clean up on error
        clear_training_state();
        throw;
    }
}

float LSTMPredictor::compute_loss(const std::vector<float>& output,
                                const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::runtime_error("Output and target size mismatch in compute_loss");
    }
    
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = output[i] - target[i];
        loss += 0.5f * diff * diff;  // MSE loss
    }
    
    return loss;
}

std::vector<float> LSTMPredictor::get_final_prediction(const LSTMOutput& lstm_output) {
    std::vector<float> final_output(num_classes, 0.0f);
    
    // Get the final hidden state directly
    const auto& final_hidden = lstm_output.sequence_output.back().back();
    
    // Compute the output
    for (int i = 0; i < num_classes; ++i) {
        float sum = fc_bias[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += fc_weight[i][j] * final_hidden[j];
        }
        final_output[i] = sum;
    }
    
    return final_output;
}

void LSTMPredictor::initialize_weights() {
    // Initialize with PyTorch's default initialization
    float k = 1.0f / std::sqrt(hidden_size);
    std::uniform_real_distribution<float> dist(-k, k);
    std::mt19937 gen(random_seed);

    // Initialize FC layer first
    fc_weight.resize(num_classes, std::vector<float>(hidden_size));
    fc_bias.resize(num_classes);
    
    // Initialize FC weights and bias
    for (int i = 0; i < num_classes; ++i) {
        fc_bias[i] = 0.0f;  // PyTorch default
        for (int j = 0; j < hidden_size; ++j) {
            fc_weight[i][j] = dist(gen);
        }
    }

    // Initialize LSTM layers
    lstm_layers.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        
        // Initialize with PyTorch dimensions
        lstm_layers[layer].weight_ih.resize(4 * hidden_size, 
            std::vector<float>(input_size_layer));
        lstm_layers[layer].weight_hh.resize(4 * hidden_size, 
            std::vector<float>(hidden_size));
        lstm_layers[layer].bias_ih.resize(4 * hidden_size);
        lstm_layers[layer].bias_hh.resize(4 * hidden_size);
        
        // Initialize weights and biases
        for (int i = 0; i < 4 * hidden_size; ++i) {
            for (int j = 0; j < input_size_layer; ++j) {
                lstm_layers[layer].weight_ih[i][j] = dist(gen);
            }
            for (int j = 0; j < hidden_size; ++j) {
                lstm_layers[layer].weight_hh[i][j] = dist(gen);
            }
            lstm_layers[layer].bias_ih[i] = dist(gen);
            lstm_layers[layer].bias_hh[i] = dist(gen);
        }
    }
}


void LSTMPredictor::set_weights(const std::vector<LSTMLayer>& weights) {
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        // Deep copy weight_ih
        lstm_layers[layer].weight_ih.resize(weights[layer].weight_ih.size());
        for (size_t i = 0; i < weights[layer].weight_ih.size(); ++i) {
            lstm_layers[layer].weight_ih[i] = weights[layer].weight_ih[i];
        }
        
        // Deep copy weight_hh
        lstm_layers[layer].weight_hh.resize(weights[layer].weight_hh.size());
        for (size_t i = 0; i < weights[layer].weight_hh.size(); ++i) {
            lstm_layers[layer].weight_hh[i] = weights[layer].weight_hh[i];
        }
        
        // Deep copy biases
        lstm_layers[layer].bias_ih = weights[layer].bias_ih;
        lstm_layers[layer].bias_hh = weights[layer].bias_hh;
        
    }
}

void LSTMPredictor::save_weights(std::ofstream& file) {
    try {
        // Save LSTM layer weights
        for (int layer = 0; layer < num_layers; ++layer) {
            // Save weight_ih dimensions and data
            size_t ih_rows = lstm_layers[layer].weight_ih.size();
            size_t ih_cols = lstm_layers[layer].weight_ih[0].size();
            file.write(reinterpret_cast<const char*>(&ih_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&ih_cols), sizeof(size_t));
            
            for (const auto& row : lstm_layers[layer].weight_ih) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }

            // Save weight_hh dimensions and data
            size_t hh_rows = lstm_layers[layer].weight_hh.size();
            size_t hh_cols = lstm_layers[layer].weight_hh[0].size();
            file.write(reinterpret_cast<const char*>(&hh_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&hh_cols), sizeof(size_t));
            
            for (const auto& row : lstm_layers[layer].weight_hh) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
        }

        // Save FC layer weights
        size_t fc_rows = fc_weight.size();
        size_t fc_cols = fc_weight[0].size();
        file.write(reinterpret_cast<const char*>(&fc_rows), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&fc_cols), sizeof(size_t));
        
        for (const auto& row : fc_weight) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving weights: " + std::string(e.what()));
    }
}

void LSTMPredictor::save_biases(std::ofstream& file) {
    try {
        // Save LSTM layer biases
        for (int layer = 0; layer < num_layers; ++layer) {
            // Save bias_ih
            size_t ih_size = lstm_layers[layer].bias_ih.size();
            file.write(reinterpret_cast<const char*>(&ih_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(lstm_layers[layer].bias_ih.data()), 
                      ih_size * sizeof(float));

            // Save bias_hh
            size_t hh_size = lstm_layers[layer].bias_hh.size();
            file.write(reinterpret_cast<const char*>(&hh_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(lstm_layers[layer].bias_hh.data()), 
                      hh_size * sizeof(float));
        }

        // Save FC layer bias
        size_t fc_size = fc_bias.size();
        file.write(reinterpret_cast<const char*>(&fc_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(fc_bias.data()), fc_size * sizeof(float));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving biases: " + std::string(e.what()));
    }
}

void LSTMPredictor::load_weights(std::ifstream& file) {
    try {
        // Load LSTM layer weights
        for (int layer = 0; layer < num_layers; ++layer) {
            // Load weight_ih
            size_t ih_rows, ih_cols;
            file.read(reinterpret_cast<char*>(&ih_rows), sizeof(size_t));
            file.read(reinterpret_cast<char*>(&ih_cols), sizeof(size_t));
            
            lstm_layers[layer].weight_ih.resize(ih_rows, std::vector<float>(ih_cols));
            for (auto& row : lstm_layers[layer].weight_ih) {
                file.read(reinterpret_cast<char*>(row.data()), ih_cols * sizeof(float));
            }

            // Load weight_hh
            size_t hh_rows, hh_cols;
            file.read(reinterpret_cast<char*>(&hh_rows), sizeof(size_t));
            file.read(reinterpret_cast<char*>(&hh_cols), sizeof(size_t));
            
            lstm_layers[layer].weight_hh.resize(hh_rows, std::vector<float>(hh_cols));
            for (auto& row : lstm_layers[layer].weight_hh) {
                file.read(reinterpret_cast<char*>(row.data()), hh_cols * sizeof(float));
            }
        }

        // Load FC layer weights
        size_t fc_rows, fc_cols;
        file.read(reinterpret_cast<char*>(&fc_rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&fc_cols), sizeof(size_t));
        
        fc_weight.resize(fc_rows, std::vector<float>(fc_cols));
        for (auto& row : fc_weight) {
            file.read(reinterpret_cast<char*>(row.data()), fc_cols * sizeof(float));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading weights: " + std::string(e.what()));
    }
}

void LSTMPredictor::load_biases(std::ifstream& file) {
    try {
        // Load LSTM layer biases
        for (int layer = 0; layer < num_layers; ++layer) {
            // Load bias_ih
            size_t ih_size;
            file.read(reinterpret_cast<char*>(&ih_size), sizeof(size_t));
            lstm_layers[layer].bias_ih.resize(ih_size);
            file.read(reinterpret_cast<char*>(lstm_layers[layer].bias_ih.data()), 
                     ih_size * sizeof(float));

            // Load bias_hh
            size_t hh_size;
            file.read(reinterpret_cast<char*>(&hh_size), sizeof(size_t));
            lstm_layers[layer].bias_hh.resize(hh_size);
            file.read(reinterpret_cast<char*>(lstm_layers[layer].bias_hh.data()), 
                     hh_size * sizeof(float));
        }

        // Load FC layer bias
        size_t fc_size;
        file.read(reinterpret_cast<char*>(&fc_size), sizeof(size_t));
        fc_bias.resize(fc_size);
        file.read(reinterpret_cast<char*>(fc_bias.data()), fc_size * sizeof(float));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading biases: " + std::string(e.what()));
    }
}

void LSTMPredictor::initialize_layer_cache() {
    if (is_layer_cache_initialized()) {
        return;  // Cache already initialized
    }
    
    std::cout << "Initializing layer cache..." << std::endl;
    
    // Initialize layer cache for each LSTM layer
    layer_cache.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        std::cout << "Initializing cache for layer " << layer << std::endl;
        
        // Initialize cache for each batch
        layer_cache[layer].resize(1);  // Single batch
        for (int batch = 0; batch < 1; ++batch) {
            // Initialize cache for each sequence step
            layer_cache[layer][batch].resize(seq_length);
            for (int seq = 0; seq < seq_length; ++seq) {
                auto& cache = layer_cache[layer][batch][seq];
                
                // Initialize all vectors with correct sizes
                int expected_input_size = (layer == 0) ? input_size : hidden_size;
                cache.input.resize(expected_input_size, 0.0f);
                cache.prev_hidden.resize(hidden_size, 0.0f);
                cache.prev_cell.resize(hidden_size, 0.0f);
                cache.cell_state.resize(hidden_size, 0.0f);
                cache.input_gate.resize(hidden_size, 0.0f);
                cache.forget_gate.resize(hidden_size, 0.0f);
                cache.cell_candidate.resize(hidden_size, 0.0f);
                cache.output_gate.resize(hidden_size, 0.0f);
                cache.hidden_state.resize(hidden_size, 0.0f);
            }
        }
    }
    
    set_is_cache_initialized(true);
    std::cout << "Layer cache initialization complete" << std::endl;
}

void LSTMPredictor::save_layer_cache(std::ofstream& file) const {
    try {
        // Save layer cache dimensions
        size_t num_batches = layer_cache.empty() ? 0 : layer_cache[0].size();
        file.write(reinterpret_cast<const char*>(&num_batches), sizeof(size_t));
        
        if (num_batches > 0) {
            size_t num_timesteps = layer_cache[0][0].size();
            file.write(reinterpret_cast<const char*>(&num_timesteps), sizeof(size_t));
            
            // Save each cache entry
            for (const auto& layer : layer_cache) {
                for (const auto& batch : layer) {
                    for (const auto& entry : batch) {
                        // Save vectors from LSTMCacheEntry
                        auto save_vector = [&file](const std::vector<float>& vec) {
                            size_t size = vec.size();
                            file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
                            if (size > 0) {
                                file.write(reinterpret_cast<const char*>(vec.data()), 
                                         size * sizeof(float));
                            }
                        };
                        
                        save_vector(entry.input);
                        save_vector(entry.prev_hidden);
                        save_vector(entry.prev_cell);
                        save_vector(entry.cell_state);
                        save_vector(entry.input_gate);
                        save_vector(entry.forget_gate);
                        save_vector(entry.cell_candidate);
                        save_vector(entry.output_gate);
                        save_vector(entry.hidden_state);
                    }
                }
            }
        }
        
        // Save h_state and c_state
        size_t state_layers = h_state.size();
        file.write(reinterpret_cast<const char*>(&state_layers), sizeof(size_t));
        for (size_t i = 0; i < state_layers; ++i) {
            size_t state_size = h_state[i].size();
            file.write(reinterpret_cast<const char*>(&state_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(h_state[i].data()), 
                      state_size * sizeof(float));
            file.write(reinterpret_cast<const char*>(c_state[i].data()), 
                      state_size * sizeof(float));
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving layer cache: " + std::string(e.what()));
    }
}

void LSTMPredictor::load_layer_cache(std::ifstream& file) {
    try {
        // Load layer cache dimensions
        size_t num_batches;
        file.read(reinterpret_cast<char*>(&num_batches), sizeof(size_t));
        
        if (num_batches > 0) {
            size_t num_timesteps;
            file.read(reinterpret_cast<char*>(&num_timesteps), sizeof(size_t));
            
            // Resize layer cache
            layer_cache.resize(num_layers);
            for (auto& layer : layer_cache) {
                layer.resize(num_batches);
                for (auto& batch : layer) {
                    batch.resize(num_timesteps);
                }
            }
            
            // Load each cache entry
            for (auto& layer : layer_cache) {
                for (auto& batch : layer) {
                    for (auto& entry : batch) {
                        // Load vectors into LSTMCacheEntry
                        auto load_vector = [&file](std::vector<float>& vec) {
                            size_t size;
                            file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
                            vec.resize(size);
                            if (size > 0) {
                                file.read(reinterpret_cast<char*>(vec.data()), 
                                        size * sizeof(float));
                            }
                        };
                        
                        load_vector(entry.input);
                        load_vector(entry.prev_hidden);
                        load_vector(entry.prev_cell);
                        load_vector(entry.cell_state);
                        load_vector(entry.input_gate);
                        load_vector(entry.forget_gate);
                        load_vector(entry.cell_candidate);
                        load_vector(entry.output_gate);
                        load_vector(entry.hidden_state);
                    }
                }
            }
        }
        
        // Load h_state and c_state
        size_t state_layers;
        file.read(reinterpret_cast<char*>(&state_layers), sizeof(size_t));
        h_state.resize(state_layers);
        c_state.resize(state_layers);
        for (size_t i = 0; i < state_layers; ++i) {
            size_t state_size;
            file.read(reinterpret_cast<char*>(&state_size), sizeof(size_t));
            h_state[i].resize(state_size);
            c_state[i].resize(state_size);
            file.read(reinterpret_cast<char*>(h_state[i].data()), 
                     state_size * sizeof(float));
            file.read(reinterpret_cast<char*>(c_state[i].data()), 
                     state_size * sizeof(float));
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading layer cache: " + std::string(e.what()));
    }
}



void LSTMPredictor::clear_training_state() {
    // Clear layer cache but maintain structure with minimal sizes
    for (auto& layer : layer_cache) {
        for (auto& batch : layer) {
            batch.resize(std::min(batch.size(), static_cast<size_t>(seq_length)));
            
            for (auto& seq : batch) {
                int expected_input_size = (current_layer == 0) ? input_size : hidden_size;
                seq.input.resize(expected_input_size);
                seq.prev_hidden.resize(hidden_size);
                seq.prev_cell.resize(hidden_size);
                seq.cell_state.resize(hidden_size);
                seq.input_gate.resize(hidden_size);
                seq.forget_gate.resize(hidden_size);
                seq.cell_candidate.resize(hidden_size);
                seq.output_gate.resize(hidden_size);
                seq.hidden_state.resize(hidden_size);
                
                std::fill(seq.input.begin(), seq.input.end(), 0.0f);
                std::fill(seq.prev_hidden.begin(), seq.prev_hidden.end(), 0.0f);
                std::fill(seq.prev_cell.begin(), seq.prev_cell.end(), 0.0f);
                std::fill(seq.cell_state.begin(), seq.cell_state.end(), 0.0f);
                std::fill(seq.input_gate.begin(), seq.input_gate.end(), 0.0f);
                std::fill(seq.forget_gate.begin(), seq.forget_gate.end(), 0.0f);
                std::fill(seq.cell_candidate.begin(), seq.cell_candidate.end(), 0.0f);
                std::fill(seq.output_gate.begin(), seq.output_gate.end(), 0.0f);
                std::fill(seq.hidden_state.begin(), seq.hidden_state.end(), 0.0f);
            }
        }
        layer.resize(1);
    }
    
    // Clear gradients used for testing but maintain structure
    for (auto& gradient : last_gradients) {
        int input_size_layer = (current_layer == 0) ? input_size : hidden_size;
        
        for (auto& row : gradient.weight_ih_grad) {
            row.resize(input_size_layer);
            std::fill(row.begin(), row.end(), 0.0f);
        }
        
        for (auto& row : gradient.weight_hh_grad) {
            row.resize(hidden_size);
            std::fill(row.begin(), row.end(), 0.0f);
        }
        
        std::fill(gradient.bias_ih_grad.begin(), gradient.bias_ih_grad.end(), 0.0f);
        std::fill(gradient.bias_hh_grad.begin(), gradient.bias_hh_grad.end(), 0.0f);
    }
    
    current_layer = 0;
    current_timestep = 0;
    current_batch = 0;
    current_cache_size = 0;
    
    // Reinitialize the cache to ensure it's properly structured
    initialize_layer_cache();
}




void LSTMPredictor::apply_sgd_update(
    std::vector<std::vector<float>>& weights,
    std::vector<std::vector<float>>& grads,
    float learning_rate,
    float momentum) {

    float max_grad = 0.0f;
    float max_update = 0.0f;
    
    // Determine which velocity terms to use based on the weights being updated
    std::vector<std::vector<float>>* velocity_terms = nullptr;
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
        throw std::runtime_error("Unknown weight dimensions in apply_sgd_update");
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            float grad = grads[i][j];
            
            // Gradient clipping
            const float GRAD_CLIP = 1.0f;
            grad = std::max(std::min(grad, GRAD_CLIP), -GRAD_CLIP);
            
            // Update velocity with momentum
            float velocity = momentum * (*velocity_terms)[i][j] - learning_rate * grad;
            (*velocity_terms)[i][j] = velocity;
            
            // Update weights using velocity
            weights[i][j] += velocity;
            
            // Track statistics
            max_grad = std::max(max_grad, std::abs(grad));
            max_update = std::max(max_update, std::abs(velocity));
        }
    }
}

void LSTMPredictor::apply_sgd_update(
    std::vector<float>& biases,
    std::vector<float>& grads,
    float learning_rate,
    float momentum) {
    
    float max_grad = 0.0f;
    float max_update = 0.0f;
    
    // Determine which velocity terms to use based on the biases being updated
    std::vector<float>* velocity_terms = nullptr;
    
    // Check if this is the FC layer bias
    if (biases.size() == num_classes) {
        velocity_terms = &velocity_fc_bias;
    }
    // Check if this is an LSTM layer bias
    else if (biases.size() == 4 * hidden_size) {
        // Find which LSTM layer this bias belongs to
        for (int layer = 0; layer < num_layers; ++layer) {
            if (&biases == &lstm_layers[layer].bias_ih) {
                velocity_terms = &velocity_bias_ih[layer];
                current_layer = layer;  // Set current layer for proper indexing
                break;
            } else if (&biases == &lstm_layers[layer].bias_hh) {
                velocity_terms = &velocity_bias_hh[layer];
                current_layer = layer;  // Set current layer for proper indexing
                break;
            }
        }
    }
    
    if (!velocity_terms) {
        throw std::runtime_error("Unknown bias dimensions in apply_sgd_update: size=" + 
                               std::to_string(biases.size()) + 
                               ", expected=" + std::to_string(num_classes) + 
                               " or " + std::to_string(4 * hidden_size));
    }
    
    for (size_t i = 0; i < biases.size(); ++i) {
        float grad = grads[i];
        
        // Optional gradient clipping
        const float GRAD_CLIP = 1.0f;
        grad = std::max(std::min(grad, GRAD_CLIP), -GRAD_CLIP);
        
        // Update velocity with momentum
        float velocity = momentum * (*velocity_terms)[i] - learning_rate * grad;
        (*velocity_terms)[i] = velocity;
        
        // Update biases using velocity
        biases[i] += velocity;
        
        // Track statistics
        max_grad = std::max(max_grad, std::abs(grad));
        max_update = std::max(max_update, std::abs(velocity));
    }
}

void LSTMPredictor::reset_layer_cache() {
    // Clear existing cache
    {
        std::vector<std::vector<std::vector<LSTMCacheEntry>>> empty;
        layer_cache.swap(empty);
    }
    
    // Initialize with minimal structure
    layer_cache.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        layer_cache[layer].resize(1);  // One batch
        layer_cache[layer][0].resize(seq_length);  // Sequence length
        
        // Initialize each cache entry with properly sized vectors
        for (auto& seq : layer_cache[layer][0]) {
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
    
    // Reset current position trackers
    current_layer = 0;
    current_timestep = 0;
    current_batch = 0;
    current_cache_size = 0;
}

void LSTMPredictor::clear_temporary_data() {
    // Clear gradients with swap
    for (auto& grad : last_gradients) {
        {
            std::vector<std::vector<float>> empty;
            grad.weight_ih_grad.swap(empty);
            grad.weight_hh_grad.swap(empty);
        }
        {
            std::vector<float> empty;
            grad.bias_ih_grad.swap(empty);
            grad.bias_hh_grad.swap(empty);
        }
    }
    
    // Reset current position trackers
    current_layer = 0;
    current_timestep = 0;
    current_batch = 0;
    current_cache_size = 0;
    
}

void LSTMPredictor::clear_layer_cache() {
    // Clear existing cache
    clear_temporary_data();
    
    // Reinitialize with minimal structure
    layer_cache.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        layer_cache[layer].resize(1);  // One batch
        layer_cache[layer][0].resize(seq_length);  // Sequence length
        
        for (auto& seq : layer_cache[layer][0]) {
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
    
    // Set cache as initialized
    set_is_cache_initialized(true);
}

void LSTMPredictor::clear_update_state() {
    // Clear existing data
    clear_temporary_data();
    
    // Reinitialize gradients with correct sizes
    last_gradients.resize(num_layers);
    for (auto& grad : last_gradients) {
        int input_size_layer = (current_layer == 0) ? input_size : hidden_size;
        grad.weight_ih_grad.resize(4 * hidden_size, std::vector<float>(input_size_layer, 0.0f));
        grad.weight_hh_grad.resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        grad.bias_ih_grad.resize(4 * hidden_size, 0.0f);
        grad.bias_hh_grad.resize(4 * hidden_size, 0.0f);
    }
}
