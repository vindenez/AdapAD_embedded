#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"

#include <random>
#include <algorithm>
#include <iostream>

#ifdef DEBUG
#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x) do {} while (0)
#endif

LSTMPredictor::LSTMPredictor(int num_classes, int input_size, int hidden_size, 
                            int num_layers, int lookback_len, 
                            bool batch_first)
    : num_classes(num_classes),
      num_layers(num_layers),
      input_size(input_size),
      hidden_size(hidden_size),
      seq_length(lookback_len),
      batch_first(batch_first) {
    
    DEBUG_PRINT(("Initializing LSTM Predictor\n"));
    DEBUG_PRINT(("Classes: %d, Input: %d, Hidden: %d, Layers: %d, Seq: %d\n",
        num_classes, input_size, hidden_size, num_layers, lookback_len));
    
    lstm_layers.resize(num_layers);
    last_gradients.resize(num_layers);
    
    DEBUG_PRINT(("Initializing weights\n"));
    initialize_weights();
    
    DEBUG_PRINT(("Resetting states\n"));
    reset_states();
    
    DEBUG_PRINT(("Initialization complete\n"));
}

void LSTMPredictor::reset_states() {
    h_state.clear();
    c_state.clear();
    h_state.resize(num_layers, std::vector<float>(hidden_size, 0.0f));
    c_state.resize(num_layers, std::vector<float>(hidden_size, 0.0f));
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
        DEBUG_PRINT(("Weight input size (%d) doesn't match expected layer input size (%d) for layer %d\n",
            weight_input_size, expected_layer_input, current_layer));
        throw std::runtime_error("Weight dimension mismatch in lstm_cell_forward");
    }

    // Verify input size
    if (input.size() != expected_layer_input) {
        DEBUG_PRINT(("Input size mismatch in layer %d: got %zu, expected %d\n",
            current_layer, input.size(), expected_layer_input));
        throw std::runtime_error("Input size mismatch in lstm_cell_forward");
    }

    // Debug prints
    DEBUG_PRINT(("  expected_layer_input: %d\n", expected_layer_input));
    DEBUG_PRINT(("  hidden_size: %d\n", hidden_size));
    DEBUG_PRINT(("  input.size(): %zu\n", input.size()));
    DEBUG_PRINT(("  h_state.size(): %zu\n", h_state.size()));
    DEBUG_PRINT(("  c_state.size(): %zu\n", c_state.size()));
    
    // Verify state dimensions
    if (h_state.size() != hidden_size) {
        DEBUG_PRINT(("Hidden state size mismatch: got %zu, expected %d\n", 
            h_state.size(), hidden_size));
        h_state.resize(hidden_size, 0.0f);
    }
    if (c_state.size() != hidden_size) {
        DEBUG_PRINT(("Cell state size mismatch: got %zu, expected %d\n", 
            c_state.size(), hidden_size));
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
    
    // **Change 1: Declare cache_entry only if training_mode is true**
    LSTMCacheEntry cache_entry;

    if (training_mode) {
        // **Change 2: Validate indices before accessing cache**
        if (current_layer >= layer_cache.size() ||
            current_batch >= layer_cache[current_layer].size() ||
            current_timestep >= layer_cache[current_layer][current_batch].size()) {
            throw std::runtime_error("Invalid cache access");
        }
        
        // **Change 3: Initialize cache entry with proper sizes**
        cache_entry = layer_cache[current_layer][current_batch][current_timestep];
        
        // Validate and copy input
        cache_entry.input = input;  // Now we know the size is correct
        
        // Initialize other cache vectors
        cache_entry.input_gate.resize(hidden_size, 0.0f);
        cache_entry.forget_gate.resize(hidden_size, 0.0f);
        cache_entry.cell_gate.resize(hidden_size, 0.0f);
        cache_entry.output_gate.resize(hidden_size, 0.0f);
        cache_entry.cell_state.resize(hidden_size, 0.0f);
        cache_entry.hidden_state.resize(hidden_size, 0.0f);
        cache_entry.prev_hidden = h_state;
        cache_entry.prev_cell = c_state;
    }
    
    // Initialize gates with biases (PyTorch layout: [i,f,g,o])
    std::vector<float> gates(4 * hidden_size);
    for (int h = 0; h < hidden_size; ++h) {
        gates[h] = layer.bias_ih[h] + layer.bias_hh[h];                     // input gate (i)
        gates[hidden_size + h] = layer.bias_ih[hidden_size + h] + 
                                layer.bias_hh[hidden_size + h];             // forget gate (f)
        gates[2 * hidden_size + h] = layer.bias_ih[2 * hidden_size + h] + 
                                    layer.bias_hh[2 * hidden_size + h];     // cell gate (g)
        gates[3 * hidden_size + h] = layer.bias_ih[3 * hidden_size + h] + 
                                    layer.bias_hh[3 * hidden_size + h];     // output gate (o)
    }
    
    // Input to hidden contributions
    for (size_t i = 0; i < input.size(); ++i) {
        for (int h = 0; h < hidden_size; ++h) {
            gates[h] += layer.weight_ih[h][i] * input[i];                     // input gate
            gates[hidden_size + h] += layer.weight_ih[hidden_size + h][i] * input[i];   // forget gate
            gates[2 * hidden_size + h] += layer.weight_ih[2 * hidden_size + h][i] * input[i]; // cell gate
            gates[3 * hidden_size + h] += layer.weight_ih[3 * hidden_size + h][i] * input[i]; // output gate
        }
    }
    
    // Hidden to hidden contributions
    for (int h = 0; h < hidden_size; ++h) {
        for (size_t i = 0; i < hidden_size; ++i) {
            gates[h] += layer.weight_hh[h][i] * h_state[i];                     // input gate
            gates[hidden_size + h] += layer.weight_hh[hidden_size + h][i] * h_state[i];   // forget gate
            gates[2 * hidden_size + h] += layer.weight_hh[2 * hidden_size + h][i] * h_state[i]; // cell gate
            gates[3 * hidden_size + h] += layer.weight_hh[3 * hidden_size + h][i] * h_state[i]; // output gate
        }
    }

    DEBUG_PRINT(("Gate values for timestep %d:\n", current_timestep));
    DEBUG_PRINT(("  input_gate[0]=%f\n", gates[0]));
    DEBUG_PRINT(("  forget_gate[0]=%f\n", gates[hidden_size]));
    DEBUG_PRINT(("  cell_gate[0]=%f\n", gates[2 * hidden_size]));
    DEBUG_PRINT(("  output_gate[0]=%f\n", gates[3 * hidden_size]));
    
    // Apply activations and update states
    for (int h = 0; h < hidden_size; ++h) {
        float i_t = sigmoid(gates[h]);                    // input gate
        float f_t = sigmoid(gates[hidden_size + h]);      // forget gate
        float g_t = tanh_custom(gates[2 * hidden_size + h]); // cell gate
        float o_t = sigmoid(gates[3 * hidden_size + h]);  // output gate

        if (h == 0) {  // Only print first unit to avoid spam
            DEBUG_PRINT(("Activated gate values:\n"));
            DEBUG_PRINT(("  i_t=%f, f_t=%f, g_t=%f, o_t=%f\n", i_t, f_t, g_t, o_t));
        }
        
        // Update cell state
        float new_cell = f_t * c_state[h] + i_t * g_t;
        c_state[h] = new_cell;
        
        // Update hidden state
        float new_hidden = o_t * tanh_custom(new_cell);
        h_state[h] = new_hidden;

        // **Change 4: Store values in cache only if training_mode is true**
        if (training_mode) {
            cache_entry.input_gate[h] = i_t;
            cache_entry.forget_gate[h] = f_t;
            cache_entry.cell_gate[h] = g_t;
            cache_entry.output_gate[h] = o_t;
            cache_entry.cell_state[h] = new_cell;
            cache_entry.hidden_state[h] = new_hidden;
        }
    }
    
    // Ensure h_state has the correct size
    if (h_state.size() != hidden_size) {
        DEBUG_PRINT(("h_state size mismatch: got %zu, expected %d\n", h_state.size(), hidden_size));
        h_state.resize(hidden_size);
    }

    // Create a copy of h_state to return
    std::vector<float> output = h_state;
    
    // Verify output dimensions one last time
    if (output.size() != hidden_size) {
        DEBUG_PRINT(("Output size mismatch: got %zu, expected %d\n", 
            output.size(), hidden_size));
        throw std::runtime_error("Output size mismatch in lstm_cell_forward");
    }

    // **Change 5: If training_mode, store cache_entry back to layer_cache**
    if (training_mode) {
        layer_cache[current_layer][current_batch][current_timestep] = cache_entry;
    }

    return output;
}


LSTMPredictor::LSTMOutput LSTMPredictor::forward(
    const std::vector<std::vector<std::vector<float>>>& x,
    const std::vector<std::vector<float>>* initial_hidden,
    const std::vector<std::vector<float>>* initial_cell) {
    
    // Detailed input validation
    DEBUG_PRINT(("Forward pass input validation:\n"));
    DEBUG_PRINT(("  batch_size = %zu\n", x.size()));
    
    for (size_t batch = 0; batch < x.size(); ++batch) {
        DEBUG_PRINT(("  Batch %zu:\n", batch));
        DEBUG_PRINT(("    sequence_length = %zu\n", x[batch].size()));
        
        for (size_t seq = 0; seq < x[batch].size(); ++seq) {
            DEBUG_PRINT(("    Timestep %zu:\n", seq));
            DEBUG_PRINT(("      features = %zu (expected %d)\n", x[batch][seq].size(), input_size));
            
            // Print first few feature values
            if (!x[batch][seq].empty()) {
                DEBUG_PRINT(("      First features: "));
                for (size_t f = 0; f < std::min(size_t(3), x[batch][seq].size()); ++f) {
                    DEBUG_PRINT(("%f ", x[batch][seq][f]));
                }
                DEBUG_PRINT(("\n"));
            }
            
            // Verify input dimensions for each timestep
            if (x[batch][seq].size() != input_size) {
                DEBUG_PRINT(("Input dimension mismatch at batch %zu, timestep %zu: got %zu features\n",
                    batch, seq, x[batch][seq].size()));
                throw std::runtime_error("Input dimension mismatch in sequence");
            }
        }
    }
    
    DEBUG_PRINT(("Starting forward pass\n"));
    DEBUG_PRINT(("Input dimensions: batch=%zu, seq=%zu, features=%zu\n",
        x.size(), x[0].size(), x[0][0].size()));
    
    try {
        size_t batch_size = x.size();
        size_t seq_len = x[0].size();
        
        // Initialize layer cache for training
        if (training_mode) {
            if (layer_cache.empty() || 
                layer_cache.size() != num_layers ||
                layer_cache[0].size() != batch_size ||
                layer_cache[0][0].size() != seq_len) {
                
                DEBUG_PRINT(("Initializing cache: layers=%d, batch=%zu, seq=%zu\n", 
                    num_layers, batch_size, seq_len));
                    
                layer_cache.clear();
                layer_cache.resize(num_layers);
                
                for (int layer = 0; layer < num_layers; ++layer) {
                    layer_cache[layer].resize(batch_size);
                    for (size_t batch = 0; batch < batch_size; ++batch) {
                        layer_cache[layer][batch].resize(seq_len);
                    }
                }
            }
        }
        
        // Initialize output structure
        LSTMOutput output;
        output.sequence_output.resize(batch_size, 
            std::vector<std::vector<float>>(seq_len, 
                std::vector<float>(hidden_size)));
        
        // Initialize or use provided states
        if (!initial_hidden || !initial_cell) {
            h_state.resize(num_layers);
            c_state.resize(num_layers);
            for (int layer = 0; layer < num_layers; ++layer) {
                h_state[layer].resize(hidden_size, 0.0f);
                c_state[layer].resize(hidden_size, 0.0f);
            }
        } else {
            h_state = *initial_hidden;
            c_state = *initial_cell;
        }
        
        // Process each batch and timestep
        for (size_t batch = 0; batch < batch_size; ++batch) {
            current_batch = batch;  // Set unconditionally
            
            if (!initial_hidden || !initial_cell) {
                reset_states();
            }
            
            for (size_t t = 0; t < seq_len; ++t) {
                current_timestep = t;
                
                std::vector<float> layer_input = x[batch][t];
                std::vector<std::vector<float>> layer_outputs(num_layers + 1);
                layer_outputs[0] = layer_input;
                
                DEBUG_PRINT(("Timestep %zu - Initial input size: %zu\n", t, layer_input.size()));
                
                // Process through LSTM layers
                for (int layer = 0; layer < num_layers; ++layer) {
                    current_layer = layer;
                    
                    // Get correct input size for this layer
                    int expected_input_size = (layer == 0) ? input_size : hidden_size;
                    
                    // Verify dimensions before forward pass
                    if (layer_outputs[layer].size() != expected_input_size) {
                        throw std::runtime_error("Layer input dimension mismatch at layer " + 
                                               std::to_string(layer));
                    }
                    
                    DEBUG_PRINT(("Layer %d before: input_size=%zu, h_state_size=%zu, c_state_size=%zu\n", 
                        layer, layer_outputs[layer].size(), 
                        h_state[layer].size(), 
                        c_state[layer].size()));
                    
                    if (training_mode) {
                        DEBUG_PRINT(("Layer %d weights - ih: %zux%zu, hh: %zux%zu\n",
                            layer,
                            lstm_layers[layer].weight_ih.size(),
                            lstm_layers[layer].weight_ih[0].size(),
                            lstm_layers[layer].weight_hh.size(),
                            lstm_layers[layer].weight_hh[0].size()));
                    }
                    
                    layer_outputs[layer + 1] = lstm_cell_forward(
                        layer_outputs[layer],
                        h_state[layer],
                        c_state[layer],
                        lstm_layers[layer]
                    );
                    
                    DEBUG_PRINT(("Layer %d after: output_size=%zu\n", layer, layer_outputs[layer + 1].size()));
                }
                
                DEBUG_PRINT(("Final output size for timestep %zu: %zu\n", 
                    t, layer_outputs[num_layers].size()));
                    
                output.sequence_output[batch][t] = layer_outputs[num_layers];
            }
        }
        
        output.final_hidden = h_state;
        output.final_cell = c_state;
        
        return output;
        
    } catch (const std::exception& e) {
        DEBUG_PRINT(("Exception in forward: %s\n", e.what()));
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
    
    DEBUG_PRINT(("backward_linear_layer - Input sizes: grad_output=%zu, last_hidden=%zu\n",
        grad_output.size(), last_hidden.size()));
    
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
    // We need to multiply fc_weight^T (4xnum_classes) with grad_output (num_classes)
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
    
    DEBUG_PRINT(("Starting backward_lstm_layer\n"));
    DEBUG_PRINT(("Gradient output: %f\n", grad_output[0]));
    DEBUG_PRINT(("cache dimensions: layers=%zu, batch=%zu, seq=%zu\n",
        cache.size(),
        cache.empty() ? 0 : cache[0].size(),
        cache.empty() || cache[0].empty() ? 0 : cache[0][0].size()));
    
    // Add dimension validation
    if (grad_output.size() != hidden_size) {
        DEBUG_PRINT(("Dimension mismatch - grad_output: %zu, hidden_size: %d\n",
            grad_output.size(), hidden_size));
        throw std::runtime_error("grad_output size mismatch in backward_lstm_layer");
    }
    
    // Add cache validation
    if (cache.size() != num_layers) {
        DEBUG_PRINT(("Cache layer mismatch - cache: %zu, num_layers: %d\n",
            cache.size(), num_layers));
        throw std::runtime_error("cache layer count mismatch in backward_lstm_layer");
    }
    
    std::vector<LSTMGradients> layer_grads(num_layers);
    
    // Initialize gradients for each layer
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        layer_grads[layer].weight_ih_grad.resize(4 * hidden_size, 
            std::vector<float>(input_size_layer, 0.0f));
        layer_grads[layer].weight_hh_grad.resize(4 * hidden_size, 
            std::vector<float>(hidden_size, 0.0f));
        layer_grads[layer].bias_ih_grad.resize(4 * hidden_size, 0.0f);
        layer_grads[layer].bias_hh_grad.resize(4 * hidden_size, 0.0f);
        DEBUG_PRINT(("Initialized gradients for layer %d: ih=%zux%zu, hh=%zux%zu, bias_ih=%zu, bias_hh=%zu\n",
            layer,
            layer_grads[layer].weight_ih_grad.size(),
            layer_grads[layer].weight_ih_grad[0].size(),
            layer_grads[layer].weight_hh_grad.size(),
            layer_grads[layer].weight_hh_grad[0].size(),
            layer_grads[layer].bias_ih_grad.size(),
            layer_grads[layer].bias_hh_grad.size()));
    }
    
    // Initialize dh_next and dc_next for each layer
    std::vector<std::vector<float>> dh_next(num_layers, std::vector<float>(hidden_size, 0.0f));
    std::vector<std::vector<float>> dc_next(num_layers, std::vector<float>(hidden_size, 0.0f));
    
    // Start from the last layer and move backward
    for (int layer = num_layers - 1; layer >= 0; --layer) {
        DEBUG_PRINT(("Processing layer %d\n", layer));
        DEBUG_PRINT(("current_batch=%zu\n", current_batch));

        std::vector<float> dh = dh_next[layer];
        std::vector<float> dc = dc_next[layer];
        
        // Add bounds checking before accessing cache
        if (current_batch >= cache[layer].size()) {
            DEBUG_PRINT(("Buffer overflow detected: current_batch=%zu, cache[%d].size()=%zu\n",
                current_batch, layer, cache[layer].size()));
            throw std::runtime_error("Cache batch index out of bounds");
        }
        
        const auto& layer_cache = cache[layer][current_batch];
        DEBUG_PRINT(("Layer cache size=%zu\n", layer_cache.size()));
        
        // If this is the last layer, add grad_output (like dy @ Wy.T in Python)
        if (layer == num_layers - 1) {
            for (int h = 0; h < hidden_size; ++h) {
                dh[h] += grad_output[h];
            }
            DEBUG_PRINT(("Added grad_output to last layer dh[0]=%f\n", dh[0]));
        }


        
        DEBUG_PRINT(("Starting backward pass through time, sequence length=%zu\n", layer_cache.size()));
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
                float di_t = dc_t * cache_entry.cell_gate[h] * cache_entry.input_gate[h] * (1.0f - cache_entry.input_gate[h]);
                float df_t = dc_t * cache_entry.prev_cell[h] * cache_entry.forget_gate[h] * (1.0f - cache_entry.forget_gate[h]);
                float dg_t = dc_t * cache_entry.input_gate[h] * (1.0f - cache_entry.cell_gate[h] * cache_entry.cell_gate[h]);
                

                // 3. Accumulate weight gradients
                int input_size_layer = (layer == 0) ? input_size : hidden_size;
                for (int j = 0; j < input_size_layer; ++j) {
                    float input_j = cache_entry.input[j];
                    layer_grads[layer].weight_ih_grad[h][j] += di_t * input_j;
                    layer_grads[layer].weight_ih_grad[hidden_size + h][j] += df_t * input_j;
                    layer_grads[layer].weight_ih_grad[2 * hidden_size + h][j] += dg_t * input_j;
                    layer_grads[layer].weight_ih_grad[3 * hidden_size + h][j] += do_t * input_j;
                }
                
                // 4. Accumulate hidden-hidden weight gradients
                for (int j = 0; j < hidden_size; ++j) {
                    float h_prev_j = cache_entry.prev_hidden[j];
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
                
                // 5. Accumulate bias gradients
                layer_grads[layer].bias_ih_grad[h] += di_t;
                layer_grads[layer].bias_ih_grad[hidden_size + h] += df_t;
                layer_grads[layer].bias_ih_grad[2 * hidden_size + h] += dg_t;
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
    DEBUG_PRINT(("Backward LSTM layer complete\n"));
    return layer_grads;
}


void LSTMPredictor::train_step(const std::vector<std::vector<std::vector<float>>>& x,
                              const std::vector<float>& target,
                              float learning_rate) {
    try {
        // Add detailed dimension checking for each sequence step
        DEBUG_PRINT(("Verifying input dimensions:\n"));
        for (size_t batch = 0; batch < x.size(); ++batch) {
            DEBUG_PRINT(("Batch %zu:\n", batch));
            for (size_t seq = 0; seq < x[batch].size(); ++seq) {
                DEBUG_PRINT(("  Sequence %zu: features=%zu\n", seq, x[batch][seq].size()));
                if (x[batch][seq].size() != input_size) {
                    DEBUG_PRINT(("Input dimension mismatch at batch %zu, seq %zu: got %zu features\n",
                        batch, seq, x[batch][seq].size()));
                    throw std::runtime_error(
                        "Input sequence dimension mismatch in train_step: batch " + 
                        std::to_string(batch) + ", seq " + std::to_string(seq));
                }
                // Print first few features to verify data
                DEBUG_PRINT(("    First features: "));
                for (size_t f = 0; f < std::min(size_t(3), x[batch][seq].size()); ++f) {
                    DEBUG_PRINT(("%f ", x[batch][seq][f]));
                }
                DEBUG_PRINT(("\n"));
            }
        }

        // Add more detailed dimension debugging
        DEBUG_PRINT(("Starting train_step with dimensions:\n"));
        DEBUG_PRINT(("  batch_size (x.size()) = %zu\n", x.size()));
        DEBUG_PRINT(("  seq_len (x[0].size()) = %zu\n", x[0].size()));
        DEBUG_PRINT(("  input_size (x[0][0].size()) = %zu\n", x[0][0].size()));
        DEBUG_PRINT(("  expected_input_size = %d\n", input_size));
        
        // Verify each sequence in the batch has correct dimensions
        for (size_t batch = 0; batch < x.size(); ++batch) {
            for (size_t seq = 0; seq < x[batch].size(); ++seq) {
                if (x[batch][seq].size() != input_size) {
                    DEBUG_PRINT(("Dimension mismatch at batch %zu, seq %zu: got %zu features\n",
                        batch, seq, x[batch][seq].size()));
                    throw std::runtime_error("Input feature size mismatch in sequence");
                }
            }
        }
        
        // Initialize Adam states if needed
        if (!are_adam_states_initialized()) {
            DEBUG_PRINT(("Initializing Adam states before first training step\n"));
            initialize_adam_states();
        }
        
        DEBUG_PRINT(("Starting train_step with batch_size=%zu, seq_len=%zu, input_size=%zu\n",
            x.size(), x[0].size(), x[0][0].size()));

        // Verify input dimensions
        if (x.empty() || x[0].empty() || x[0][0].empty()) {
            DEBUG_PRINT(("Empty input tensor in train_step\n"));
            throw std::runtime_error("Empty input tensor");
        }
        if (x[0][0].size() != input_size) {
            DEBUG_PRINT(("Input feature size mismatch: got %zu, expected %d\n",
                x[0][0].size(), input_size));
            throw std::runtime_error("Input feature size mismatch");
        }
        
        // Verify target dimensions
        if (target.size() != num_classes) {
            throw std::invalid_argument("Target size mismatch");
        }
        
        // Adam hyperparameters
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        static int timestep = 0;
        timestep++;

        // Forward pass
        DEBUG_PRINT(("Starting forward pass\n"));
        auto lstm_output = forward(x);
        auto output = get_final_prediction(lstm_output);
        DEBUG_PRINT(("Forward pass complete, output size=%zu\n", output.size()));

        // Compute gradients
        auto grad_output = compute_mse_loss_gradient(output, target);
        DEBUG_PRINT(("Loss gradient computed, size=%zu, target size=%zu\n", 
            grad_output.size(), target.size()));

        // Extract final hidden state
        const auto& last_hidden = lstm_output.final_hidden.back();
        DEBUG_PRINT(("Last hidden state size=%zu\n", last_hidden.size()));

        // Backward pass through linear layer
        std::vector<std::vector<float>> fc_weight_grad;
        std::vector<float> fc_bias_grad;
        std::vector<float> lstm_grad;
        DEBUG_PRINT(("Starting linear layer backward pass\n"));
        backward_linear_layer(grad_output, last_hidden, fc_weight_grad, fc_bias_grad, lstm_grad);
        DEBUG_PRINT(("Linear backward complete\n"));

        // Verify FC layer dimensions before Adam updates
        if (fc_weight.size() != fc_weight_grad.size() || 
            fc_weight[0].size() != fc_weight_grad[0].size() ||
            fc_weight.size() != m_fc_weight.size() ||
            fc_weight[0].size() != m_fc_weight[0].size()) {
            
            DEBUG_PRINT(("FC dimension mismatch:\n"));
            DEBUG_PRINT(("fc_weight: %zux%zu\n", fc_weight.size(), fc_weight[0].size()));
            DEBUG_PRINT(("fc_weight_grad: %zux%zu\n", fc_weight_grad.size(), fc_weight_grad[0].size()));
            DEBUG_PRINT(("m_fc_weight: %zux%zu\n", m_fc_weight.size(), m_fc_weight[0].size()));
            throw std::runtime_error("Dimension mismatch in FC layer Adam update");
        }

        // Apply Adam updates to FC layer
        try {
            DEBUG_PRINT(("Starting FC Adam updates\n"));
            apply_adam_update(fc_weight, fc_weight_grad, m_fc_weight, v_fc_weight,
                            learning_rate, beta1, beta2, epsilon, timestep);
            DEBUG_PRINT(("FC weight Adam update complete\n"));
            
            apply_adam_update(fc_bias, fc_bias_grad, m_fc_bias, v_fc_bias,
                            learning_rate, beta1, beta2, epsilon, timestep);
            DEBUG_PRINT(("FC bias Adam update complete\n"));
        } catch (const std::exception& e) {
            DEBUG_PRINT(("Exception in FC Adam update: %s\n", e.what()));
            throw;
        }

        // Validate cache before LSTM backward pass
        if (layer_cache.empty()) {
            throw std::runtime_error("Empty layer cache");
        }

        // Verify lstm_grad dimensions
        if (lstm_grad.size() != hidden_size) {
            DEBUG_PRINT(("lstm_grad size: %zu, expected: %d\n", lstm_grad.size(), hidden_size));
            throw std::runtime_error("Invalid lstm_grad dimensions");
        }

        // LSTM backward pass
        DEBUG_PRINT(("Starting LSTM backward pass\n"));
        auto lstm_grads = backward_lstm_layer(lstm_grad, layer_cache, learning_rate);
        DEBUG_PRINT(("LSTM backward complete\n"));

        // Apply Adam updates to LSTM layers
        for (int layer = 0; layer < num_layers; ++layer) {
            try {
                // Verify LSTM layer dimensions before updates
                if (lstm_layers[layer].weight_ih.size() != lstm_grads[layer].weight_ih_grad.size()) {
                    throw std::runtime_error("LSTM weight_ih dimension mismatch at layer " + 
                                           std::to_string(layer));
                }
                
                const int gates_count = 4;  // input, forget, cell, output gates


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


            } catch (const std::exception& e) {
                DEBUG_PRINT(("Exception in LSTM layer %d Adam update: %s\n", layer, e.what()));
                throw;
            }
        }
        DEBUG_PRINT(("Adam updates complete\n"));

    } catch (const std::exception& e) {
        DEBUG_PRINT(("Exception in train_step: %s\n", e.what()));
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
        DEBUG_PRINT(("Loss computation: output=%f, target=%f, diff=%f, loss=%f\n",
            output[i], target[i], diff, loss));
    }
    
    return loss;
}

std::vector<float> LSTMPredictor::get_final_prediction(const LSTMOutput& lstm_output) {
    std::vector<float> final_output(num_classes, 0.0f);
    const auto& final_hidden = lstm_output.sequence_output.back().back();
    
    for (int i = 0; i < num_classes; ++i) {
        final_output[i] = fc_bias[i];
        for (int j = 0; j < hidden_size; ++j) {
            final_output[i] += fc_weight[i][j] * final_hidden[j];
        }
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

        // Add debug statements to verify dimensions
        DEBUG_PRINT(("Layer %d weight_ih dimensions: %zux%zu\n", layer,
            lstm_layers[layer].weight_ih.size(), lstm_layers[layer].weight_ih[0].size()));
        DEBUG_PRINT(("Layer %d weight_hh dimensions: %zux%zu\n", layer,
            lstm_layers[layer].weight_hh.size(), lstm_layers[layer].weight_hh[0].size()));
    }
}


void LSTMPredictor::initialize_adam_states() {
    DEBUG_PRINT(("Starting Adam state initialization\n"));
    float k = 0.0f;
    
    try {
        // Create new FC layer vectors
        m_fc_weight = std::vector<std::vector<float>>(
            num_classes, std::vector<float>(hidden_size, k));
        v_fc_weight = std::vector<std::vector<float>>(
            num_classes, std::vector<float>(hidden_size, k));
        m_fc_bias = std::vector<float>(num_classes, k);
        v_fc_bias = std::vector<float>(num_classes, k);
        
        // Initialize LSTM layer states directly
        std::vector<std::vector<std::vector<float>>> new_m_weight_ih(num_layers);
        std::vector<std::vector<std::vector<float>>> new_v_weight_ih(num_layers);
        std::vector<std::vector<std::vector<float>>> new_m_weight_hh(num_layers);
        std::vector<std::vector<std::vector<float>>> new_v_weight_hh(num_layers);
        std::vector<std::vector<float>> new_m_bias_ih(num_layers);
        std::vector<std::vector<float>> new_v_bias_ih(num_layers);
        std::vector<std::vector<float>> new_m_bias_hh(num_layers);
        std::vector<std::vector<float>> new_v_bias_hh(num_layers);
        
        // Initialize each layer's states
        for (int layer = 0; layer < num_layers; ++layer) {
            int input_size_layer = (layer == 0) ? input_size : hidden_size;
            
            DEBUG_PRINT(("Initializing LSTM layer %d Adam states\n", layer));
            
            new_m_weight_ih[layer] = std::vector<std::vector<float>>(
                4 * hidden_size, std::vector<float>(input_size_layer, k));
            new_v_weight_ih[layer] = std::vector<std::vector<float>>(
                4 * hidden_size, std::vector<float>(input_size_layer, k));
            new_m_weight_hh[layer] = std::vector<std::vector<float>>(
                4 * hidden_size, std::vector<float>(hidden_size, k));
            new_v_weight_hh[layer] = std::vector<std::vector<float>>(
                4 * hidden_size, std::vector<float>(hidden_size, k));
            new_m_bias_ih[layer] = std::vector<float>(4 * hidden_size, k);
            new_v_bias_ih[layer] = std::vector<float>(4 * hidden_size, k);
            new_m_bias_hh[layer] = std::vector<float>(4 * hidden_size, k);
            new_v_bias_hh[layer] = std::vector<float>(4 * hidden_size, k);
            
            DEBUG_PRINT(("Layer %d weight_ih dimensions: %zux%zu\n", 
                layer, new_m_weight_ih[layer].size(), new_m_weight_ih[layer][0].size()));
        }

        // Assign new vectors to member variables
        m_weight_ih = std::move(new_m_weight_ih);
        v_weight_ih = std::move(new_v_weight_ih);
        m_weight_hh = std::move(new_m_weight_hh);
        v_weight_hh = std::move(new_v_weight_hh);
        m_bias_ih = std::move(new_m_bias_ih);
        v_bias_ih = std::move(new_v_bias_ih);
        m_bias_hh = std::move(new_m_bias_hh);
        v_bias_hh = std::move(new_v_bias_hh);

        adam_initialized = true;
        DEBUG_PRINT(("Adam initialization complete\n"));
        
    } catch (const std::exception& e) {
        DEBUG_PRINT(("Exception during Adam initialization: %s\n", e.what()));
        throw;
    }
}


void LSTMPredictor::apply_adam_update(
    std::vector<std::vector<float>>& weights, 
    std::vector<std::vector<float>>& grads,
    std::vector<std::vector<float>>& m_t, 
    std::vector<std::vector<float>>& v_t,
    float learning_rate, float beta1, float beta2, float epsilon, int t) {
    
    // Check for empty vectors
    if (weights.empty() || grads.empty() || m_t.empty() || v_t.empty()) {
        throw std::runtime_error("Empty vectors in Adam update");
    }

    // Check for empty inner vectors
    if (weights[0].empty() || grads[0].empty() || m_t[0].empty() || v_t[0].empty()) {
        throw std::runtime_error("Empty inner vectors in Adam update");
    }

    // Existing dimension checks...
    if (weights.size() != grads.size() || 
        weights[0].size() != grads[0].size() ||
        weights.size() != m_t.size() ||
        weights[0].size() != m_t[0].size() ||
        weights.size() != v_t.size() ||
        weights[0].size() != v_t[0].size()) {
        
        throw std::runtime_error("Dimension mismatch in Adam update");
    }
    
    if (t <= 0) {
        throw std::invalid_argument("Adam timestep must be positive");
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            // Update biased moments
            m_t[i][j] = beta1 * m_t[i][j] + (1.0f - beta1) * grads[i][j];
            v_t[i][j] = beta2 * v_t[i][j] + (1.0f - beta2) * grads[i][j] * grads[i][j];
            
            // Compute bias-corrected moments
            float m_hat = m_t[i][j] / (1.0f - std::pow(beta1, t));
            float v_hat = v_t[i][j] / (1.0f - std::pow(beta2, t));
            
            // Update weights
            weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            
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
    
    for (size_t i = 0; i < biases.size(); ++i) {
        // Update biased moments
        m_t[i] = beta1 * m_t[i] + (1.0f - beta1) * grads[i];
        v_t[i] = beta2 * v_t[i] + (1.0f - beta2) * grads[i] * grads[i];
        
        // Compute bias-corrected moments
        float m_hat = m_t[i] / (1.0f - std::pow(beta1, t));
        float v_hat = v_t[i] / (1.0f - std::pow(beta2, t));
        
        // Update biases
        biases[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        
    }
}

bool LSTMPredictor::are_adam_states_initialized() const {
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

    // Check dimensions
    if (m_fc_weight.size() != num_classes || 
        m_fc_weight[0].size() != hidden_size) {
        return false;
    }

    // Check LSTM layer dimensions
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        
        if (m_weight_ih[layer].size() != 4 * hidden_size ||
            m_weight_ih[layer][0].size() != input_size_layer ||
            m_weight_hh[layer].size() != 4 * hidden_size ||
            m_weight_hh[layer][0].size() != hidden_size ||
            m_bias_ih[layer].size() != 4 * hidden_size ||
            m_bias_hh[layer].size() != 4 * hidden_size) {
            return false;
        }
    }

    return adam_initialized;
}

void LSTMPredictor::set_weights(const std::vector<LSTMLayer>& weights) {
    DEBUG_PRINT(("Setting weights:\n"));
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
        
        // Debug print first few weights to verify
        if (layer == 0) {
            DEBUG_PRINT(("  After setting - Layer %zu first gate weights:\n", layer));
            DEBUG_PRINT(("    input gate: ih=%f, hh=%f\n", 
                lstm_layers[layer].weight_ih[0][0], 
                lstm_layers[layer].weight_hh[0][0]));
            DEBUG_PRINT(("    forget gate: ih=%f, hh=%f\n", 
                lstm_layers[layer].weight_ih[4][0], 
                lstm_layers[layer].weight_hh[4][0]));
        }
    }
}

