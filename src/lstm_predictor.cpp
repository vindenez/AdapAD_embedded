#include "lstm_predictor.hpp"
#include "config.hpp"
#include "matrix_utils.hpp"

#include <algorithm>
#include <cxxabi.h>
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <random>

LSTMPredictor::LSTMPredictor(int num_classes, int input_size, int hidden_size, int num_layers,
                             int lookback_len, bool batch_first)
    : num_classes(num_classes), num_layers(num_layers), input_size(input_size),
      hidden_size(hidden_size), seq_length(lookback_len), batch_first(batch_first),
      training_mode(false), online_learning_mode(false), is_cache_initialized(false) {

    std::cout << "Initializing LSTM Predictor with:" << std::endl;
    std::cout << "- num_classes: " << num_classes << std::endl;
    std::cout << "- input_size: " << input_size << std::endl;
    std::cout << "- hidden_size: " << hidden_size << std::endl;
    std::cout << "- num_layers: " << num_layers << std::endl;
    std::cout << "- lookback_len: " << lookback_len << std::endl;
    std::cout << "- random_seed: " << random_seed << std::endl;

    const Config &config = Config::getInstance();

    // Set random seed before any initialization
    set_random_seed(random_seed);

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
        layer_cache[layer].resize(1);               // One batch
        layer_cache[layer][0].resize(lookback_len); // Sequence length

        // Pre-allocate each cache entry with properly sized vectors
        for (LSTMCacheEntry &seq : layer_cache[layer][0]) {
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
    for (LSTMGradients &grad : last_gradients) {
        int input_size_layer = (current_layer == 0) ? input_size : hidden_size;
        grad.weight_ih_grad.resize(4 * hidden_size, std::vector<float>(input_size_layer, 0.0f));
        grad.weight_hh_grad.resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        grad.bias_ih_grad.resize(4 * hidden_size, 0.0f);
        grad.bias_hh_grad.resize(4 * hidden_size, 0.0f);
    }

    // Pre-allocate gradient buffers
    grad_output_buffer.resize(num_classes, 0.0f);
    fc_weight_grad_buffer.resize(num_classes, std::vector<float>(hidden_size, 0.0f));
    fc_bias_grad_buffer.resize(num_classes, 0.0f);
    lstm_grad_buffer.resize(hidden_size, 0.0f);

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

float LSTMPredictor::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float LSTMPredictor::tanh(float x) { return std::tanh(x); }

std::vector<float> LSTMPredictor::forward_lstm_cell(const std::vector<float> &input,
                                                    std::vector<float> &h_state,
                                                    std::vector<float> &c_state,
                                                    const LSTMLayer &layer) {

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
    if (layer.weight_hh.size() != 4 * hidden_size || layer.weight_hh[0].size() != hidden_size) {
        throw std::runtime_error("Weight hh dimension mismatch");
    }

    // Reference to cache entry if in training or online learning mode
    LSTMCacheEntry *cache_entry = nullptr;
    if (training_mode) { // This includes both training and online learning
        // Validate indices before accessing cache
        if (current_layer >= layer_cache.size() ||
            current_batch >= layer_cache[current_layer].size() ||
            current_timestep >= layer_cache[current_layer][current_batch].size()) {
            throw std::runtime_error("Invalid cache access");
        }

        cache_entry = &layer_cache[current_layer][current_batch][current_timestep];

        // Resize vectors to exact needed size
        cache_entry->input.resize(weight_input_size);
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
        gates[h] = layer.bias_ih[h] + layer.bias_hh[h];                         // input gate (i)
        gates[hidden_size + h] =
            layer.bias_ih[hidden_size + h] + layer.bias_hh[hidden_size + h];    // forget gate (f)
        gates[2 * hidden_size + h] = layer.bias_ih[2 * hidden_size + h] +
                                     layer.bias_hh[2 * hidden_size + h];        // cell candidate (Ĉ)
        gates[3 * hidden_size + h] = layer.bias_ih[3 * hidden_size + h] +
                                     layer.bias_hh[3 * hidden_size + h];        // output gate (o)
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
        float cell_candidate = tanh(gates[2 * hidden_size + h]);                         // cell candidate
    float o_t = sigmoid(gates[3 * hidden_size + h]);                                            // output gate

        // Update cell state
        float new_cell = f_t * c_state[h] + i_t * cell_candidate;
        c_state[h] = new_cell;

        // Update hidden state
        float new_hidden = o_t * tanh(new_cell);
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

LSTMPredictor::LSTMOutput LSTMPredictor::forward_lstm(const std::vector<std::vector<std::vector<float>>> &x,
                                                    const std::vector<std::vector<float>> *initial_hidden,
                                                    const std::vector<std::vector<float>> *initial_cell) {

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
        for (std::vector<std::vector<float>> &batch : output.sequence_output) {
            batch.resize(seq_len);
            for (std::vector<float> &seq : batch) {
                seq.resize(hidden_size, 0.0f);
            }
        }

        // Process each batch
        for (size_t batch = 0; batch < batch_size; ++batch) {
            current_batch = batch;

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

                        LSTMCacheEntry &cache_entry =
                            layer_cache[current_layer][current_batch][current_timestep];
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

                output.sequence_output[batch][t] = layer_outputs[num_layers];
            }
        }

        output.final_hidden = h_state;
        output.final_cell = c_state;

        return output;

    } catch (const std::exception &e) {
        clear_update_state();
        throw;
    }
}

std::vector<float> LSTMPredictor::forward(const std::vector<std::vector<std::vector<float>>> &x,
                          const std::vector<std::vector<float>> *initial_hidden,
                          const std::vector<std::vector<float>> *initial_cell) {
    // First process through LSTM layers
    LSTMOutput lstm_output = forward_lstm(x, initial_hidden, initial_cell);
    
    // Then process through fully connected layer
    return forward_linear(lstm_output);
}

void LSTMPredictor::backward_linear_layer(const std::vector<float> &grad_output,
                                          const std::vector<float> &last_hidden,
                                          std::vector<std::vector<float>> &weight_grad,
                                          std::vector<float> &bias_grad,
                                          std::vector<float> &input_grad) {

    // Check dimensions
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
    const std::vector<float> &grad_output,
    const std::vector<std::vector<std::vector<LSTMCacheEntry>>> &cache, float learning_rate) {

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

    // Start from the last layer and move backward
    for (int layer = num_layers - 1; layer >= 0; --layer) {

        std::vector<float> dh = dh_next[layer];
        std::vector<float> dc = dc_next[layer];

        // Add bounds checking before accessing cache
        if (current_batch >= cache[layer].size()) {
            throw std::runtime_error("Cache batch index out of bounds");
        }

        const std::vector<LSTMCacheEntry> &layer_cache = cache[layer][current_batch];

        // If this is the last layer, add grad output (like dy @ Wy.T in PyTorch)
        if (layer == num_layers - 1) {
            for (int h = 0; h < hidden_size; ++h) {
                dh[h] += grad_output[h];
            }
        }

        // Process each time step in reverse order
        for (int t = layer_cache.size() - 1; t >= 0; --t) {

            const LSTMCacheEntry &cache_entry = layer_cache[t];

            std::vector<float> dh_prev(hidden_size, 0.0f);
            std::vector<float> dc_prev(hidden_size, 0.0f);

            // Process each hidden unit
            for (int h = 0; h < hidden_size; ++h) {
                float tanh_c = tanh(cache_entry.cell_state[h]);
                float dho = dh[h];

                // 1. Cell state gradient
                float dc_t = dho * cache_entry.output_gate[h] * (1.0f - tanh_c * tanh_c);
                dc_t += dc[h]; // Add gradient from future timestep

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

float LSTMPredictor::train_step(const std::vector<std::vector<std::vector<float>>> &x,
                               const std::vector<float> &target,
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
        
        // Forward pass through LSTM layers
        LSTMOutput lstm_output = forward_lstm(x);
        
        // Get final prediction
        std::vector<float> output = forward_linear(lstm_output);

        // Calculate loss
        float loss = mse_loss(output, target);
        
        // Compute gradients (reuse pre-allocated buffer)
        mse_loss_gradient(output, target, grad_output_buffer);  
        
        // Extract final hidden state
        const std::vector<float> &last_hidden = lstm_output.sequence_output.back().back();
        
        // Backward pass through linear layer (using pre-allocated buffers)
        backward_linear_layer(grad_output_buffer, last_hidden, 
                            fc_weight_grad_buffer, fc_bias_grad_buffer, lstm_grad_buffer);
        
        // Update FC layer weights using SGD 
        apply_sgd_update(fc_weight, fc_weight_grad_buffer, learning_rate, 0.9f);
        apply_sgd_update(fc_bias, fc_bias_grad_buffer, learning_rate, 0.9f);
        
        // Get LSTM layer gradients
        std::vector<LSTMGradients> lstm_gradients = backward_lstm_layer(lstm_grad_buffer, layer_cache, learning_rate);

        // Update LSTM layer parameters using SGD 
        for (int layer = 0; layer < num_layers; ++layer) {
            if (layer >= lstm_layers.size()) {
                throw std::runtime_error("Layer index out of bounds: " + std::to_string(layer));
            }
            if (layer >= lstm_gradients.size()) {
                throw std::runtime_error("Gradient index out of bounds: " + std::to_string(layer));
            }

            apply_sgd_update(lstm_layers[layer].weight_ih, lstm_gradients[layer].weight_ih_grad,
                             learning_rate, 0.9f);
            apply_sgd_update(lstm_layers[layer].weight_hh, lstm_gradients[layer].weight_hh_grad,
                             learning_rate, 0.9f);
            apply_sgd_update(lstm_layers[layer].bias_ih, lstm_gradients[layer].bias_ih_grad,
                             learning_rate, 0.9f);
            apply_sgd_update(lstm_layers[layer].bias_hh, lstm_gradients[layer].bias_hh_grad,
                             learning_rate, 0.9f);
        }

        clear_update_state();

        return loss;

    } catch (const std::exception &e) {
        clear_update_state();
        throw;
    }
}

std::vector<float> LSTMPredictor::forward_linear(const LSTMOutput &lstm_output) {
    std::vector<float> final_output(num_classes, 0.0f);

    // Get the final hidden state directly
    const std::vector<float> &final_hidden = lstm_output.sequence_output.back().back();

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
    std::mt19937 gen(get_random_seed());

    // Initialize FC layer first
    fc_weight.resize(num_classes, std::vector<float>(hidden_size));
    fc_bias.resize(num_classes);

    // Initialize FC weights and bias
    for (int i = 0; i < num_classes; ++i) {
        fc_bias[i] = 0.0f; // PyTorch default
        for (int j = 0; j < hidden_size; ++j) {
            fc_weight[i][j] = dist(gen);
        }
    }

    // Initialize LSTM layers
    lstm_layers.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;

        // Initialize with PyTorch dimensions
        lstm_layers[layer].weight_ih.resize(4 * hidden_size, std::vector<float>(input_size_layer));
        lstm_layers[layer].weight_hh.resize(4 * hidden_size, std::vector<float>(hidden_size));
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

void LSTMPredictor::set_weights(const std::vector<LSTMLayer> &weights) {
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

void LSTMPredictor::save_weights(std::ofstream &file) {
    try {
        // Save LSTM layer weights
        for (int layer = 0; layer < num_layers; ++layer) {
            // Save weight_ih dimensions and data
            size_t ih_rows = lstm_layers[layer].weight_ih.size();
            size_t ih_cols = lstm_layers[layer].weight_ih[0].size();
            file.write(reinterpret_cast<const char *>(&ih_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char *>(&ih_cols), sizeof(size_t));

            for (const std::vector<float> &row : lstm_layers[layer].weight_ih) {
                file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
            }

            // Save weight_hh dimensions and data
            size_t hh_rows = lstm_layers[layer].weight_hh.size();
            size_t hh_cols = lstm_layers[layer].weight_hh[0].size();
            file.write(reinterpret_cast<const char *>(&hh_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char *>(&hh_cols), sizeof(size_t));

            for (const std::vector<float> &row : lstm_layers[layer].weight_hh) {
                file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
            }
        }

        // Save FC layer weights
        size_t fc_rows = fc_weight.size();
        size_t fc_cols = fc_weight[0].size();
        file.write(reinterpret_cast<const char *>(&fc_rows), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(&fc_cols), sizeof(size_t));

        for (const std::vector<float> &row : fc_weight) {
            file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error saving weights: " + std::string(e.what()));
    }
}

void LSTMPredictor::save_biases(std::ofstream &file) {
    try {
        // Save LSTM layer biases
        for (int layer = 0; layer < num_layers; ++layer) {
            // Save bias_ih
            size_t ih_size = lstm_layers[layer].bias_ih.size();
            file.write(reinterpret_cast<const char *>(&ih_size), sizeof(size_t));
            file.write(reinterpret_cast<const char *>(lstm_layers[layer].bias_ih.data()),
                       ih_size * sizeof(float));

            // Save bias_hh
            size_t hh_size = lstm_layers[layer].bias_hh.size();
            file.write(reinterpret_cast<const char *>(&hh_size), sizeof(size_t));
            file.write(reinterpret_cast<const char *>(lstm_layers[layer].bias_hh.data()),
                       hh_size * sizeof(float));
        }

        // Save FC layer bias
        size_t fc_size = fc_bias.size();
        file.write(reinterpret_cast<const char *>(&fc_size), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(fc_bias.data()), fc_size * sizeof(float));
    } catch (const std::exception &e) {
        throw std::runtime_error("Error saving biases: " + std::string(e.what()));
    }
}

void LSTMPredictor::load_weights(std::ifstream &file) {
    try {
        // Load LSTM layer weights
        for (int layer = 0; layer < num_layers; ++layer) {
            // Load weight_ih
            size_t ih_rows, ih_cols;
            file.read(reinterpret_cast<char *>(&ih_rows), sizeof(size_t));
            file.read(reinterpret_cast<char *>(&ih_cols), sizeof(size_t));

            lstm_layers[layer].weight_ih.resize(ih_rows, std::vector<float>(ih_cols));
            for (std::vector<float> &row : lstm_layers[layer].weight_ih) {
                file.read(reinterpret_cast<char *>(row.data()), ih_cols * sizeof(float));
            }

            // Load weight_hh
            size_t hh_rows, hh_cols;
            file.read(reinterpret_cast<char *>(&hh_rows), sizeof(size_t));
            file.read(reinterpret_cast<char *>(&hh_cols), sizeof(size_t));

            lstm_layers[layer].weight_hh.resize(hh_rows, std::vector<float>(hh_cols));
            for (std::vector<float> &row : lstm_layers[layer].weight_hh) {
                file.read(reinterpret_cast<char *>(row.data()), hh_cols * sizeof(float));
            }
        }

        // Load FC layer weights
        size_t fc_rows, fc_cols;
        file.read(reinterpret_cast<char *>(&fc_rows), sizeof(size_t));
        file.read(reinterpret_cast<char *>(&fc_cols), sizeof(size_t));

        fc_weight.resize(fc_rows, std::vector<float>(fc_cols));
        for (std::vector<float> &row : fc_weight) {
            file.read(reinterpret_cast<char *>(row.data()), fc_cols * sizeof(float));
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error loading weights: " + std::string(e.what()));
    }
}

void LSTMPredictor::load_biases(std::ifstream &file) {
    try {
        // Load LSTM layer biases
        for (int layer = 0; layer < num_layers; ++layer) {
            // Load bias_ih
            size_t ih_size;
            file.read(reinterpret_cast<char *>(&ih_size), sizeof(size_t));
            lstm_layers[layer].bias_ih.resize(ih_size);
            file.read(reinterpret_cast<char *>(lstm_layers[layer].bias_ih.data()),
                      ih_size * sizeof(float));

            // Load bias_hh
            size_t hh_size;
            file.read(reinterpret_cast<char *>(&hh_size), sizeof(size_t));
            lstm_layers[layer].bias_hh.resize(hh_size);
            file.read(reinterpret_cast<char *>(lstm_layers[layer].bias_hh.data()),
                      hh_size * sizeof(float));
        }

        // Load FC layer bias
        size_t fc_size;
        file.read(reinterpret_cast<char *>(&fc_size), sizeof(size_t));
        fc_bias.resize(fc_size);
        file.read(reinterpret_cast<char *>(fc_bias.data()), fc_size * sizeof(float));
    } catch (const std::exception &e) {
        throw std::runtime_error("Error loading biases: " + std::string(e.what()));
    }
}

void LSTMPredictor::initialize_layer_cache() {
    if (is_layer_cache_initialized()) {
        return;
    }

    std::cout << "Initializing layer cache..." << std::endl;

    // Initialize layer cache for each LSTM layer
    layer_cache.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        std::cout << "Initializing cache for layer " << layer << std::endl;

        // Initialize cache for each batch
        layer_cache[layer].resize(1); // Single batch
        for (int batch = 0; batch < 1; ++batch) {
            // Initialize cache for each sequence step
            layer_cache[layer][batch].resize(seq_length);
            for (int seq = 0; seq < seq_length; ++seq) {
                LSTMCacheEntry &cache = layer_cache[layer][batch][seq];

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

void LSTMPredictor::clear_update_state() {
    for (LSTMGradients &gradient : last_gradients) {
        for (std::vector<float> &row : gradient.weight_ih_grad) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
        for (std::vector<float> &row : gradient.weight_hh_grad) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
        std::fill(gradient.bias_ih_grad.begin(), gradient.bias_ih_grad.end(), 0.0f);
        std::fill(gradient.bias_hh_grad.begin(), gradient.bias_hh_grad.end(), 0.0f);


    }
    // Reset position trackers
    current_layer = 0;
    current_timestep = 0;
    current_batch = 0;
}

void LSTMPredictor::apply_sgd_update(
    std::vector<std::vector<float>>& weights,
    std::vector<std::vector<float>>& grads,
    float learning_rate,
    float momentum) {

    // Find which velocity terms to use based on dimensions
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
        // Fallback: try to find the right velocity terms by comparing addresses
        for (int layer = 0; layer < num_layers; ++layer) {
            if (&weights == &lstm_layers[layer].weight_ih) {
                velocity_terms = &velocity_weight_ih[layer];
                current_layer = layer;
                break;
            } else if (&weights == &lstm_layers[layer].weight_hh) {
                velocity_terms = &velocity_weight_hh[layer];
                current_layer = layer;
                break;
            }
        }
    }

    if (!velocity_terms) {
        throw std::runtime_error("Unknown weight dimensions in apply_sgd_update");
    }

    // Apply SGD with momentum
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            float grad = grads[i][j];
            
            // Gradient clipping
            grad = std::max(std::min(grad, 1.0f), -1.0f);
            
            // Update velocity with momentum: v = momentum * v - lr * grad
            float velocity = momentum * (*velocity_terms)[i][j] - learning_rate * grad;
            (*velocity_terms)[i][j] = velocity;
            
            // Update weights using velocity: w = w + v
            weights[i][j] += velocity;
        }
    }
}

void LSTMPredictor::apply_sgd_update(
    std::vector<float>& biases,
    std::vector<float>& grads,
    float learning_rate,
    float momentum) {
    
    // Find which velocity terms to use
    std::vector<float>* velocity_terms = nullptr;
    
    if (biases.size() == num_classes) {
        velocity_terms = &velocity_fc_bias;
    } else if (biases.size() == 4 * hidden_size) {
        // Try to find the correct bias vector
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
        throw std::runtime_error("Unknown bias dimensions in apply_sgd_update");
    }
    
    // Apply SGD with momentum
    for (size_t i = 0; i < biases.size(); ++i) {
        float grad = grads[i];
        
        // Gradient clipping
        grad = std::max(std::min(grad, 1.0f), -1.0f);
        
        // Update velocity with momentum: v = momentum * v - lr * grad
        float velocity = momentum * (*velocity_terms)[i] - learning_rate * grad;
        (*velocity_terms)[i] = velocity;
        
        // Update biases using velocity: b = b + v
        biases[i] += velocity;
    }
}


void LSTMPredictor::mse_loss_gradient(const std::vector<float> &output,
                                   const std::vector<float> &target,
                                   std::vector<float> &gradient) {
    // Ensure gradient is properly sized
    if (gradient.size() != output.size()) {
        gradient.resize(output.size());
    }
    
    for (std::size_t i = 0; i < output.size(); ++i) {
        gradient[i] = 2.0f * (output[i] - target[i]) / output.size();
    }
}

float LSTMPredictor::mse_loss(const std::vector<float> &prediction,
                              const std::vector<float> &target) {
    float loss = 0.0f;
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        float diff = prediction[i] - target[i];
        loss += diff * diff;
    }
    return loss / prediction.size();
}

void LSTMPredictor::set_lstm_weights(int layer, const std::vector<std::vector<float>> &w_ih,
                                     const std::vector<std::vector<float>> &w_hh) {
    if (layer < num_layers) {
        lstm_layers[layer].weight_ih = w_ih;
        lstm_layers[layer].weight_hh = w_hh;
    }
}

void LSTMPredictor::set_lstm_bias(int layer, const std::vector<float> &b_ih,
                                  const std::vector<float> &b_hh) {
    if (layer < num_layers) {
        lstm_layers[layer].bias_ih = b_ih;
        lstm_layers[layer].bias_hh = b_hh;
    }
}

void LSTMPredictor::set_fc_weights(const std::vector<std::vector<float>> &weights,
                                   const std::vector<float> &bias) {
    fc_weight = weights;
    fc_bias = bias;
}

void LSTMPredictor::pre_allocate_vectors(
    std::vector<std::vector<std::vector<float>>>& input,
    std::vector<float>& target,
    std::vector<float>& pred,
    LSTMOutput& output,
    int batch_size,
    int seq_len,
    int target_size
) {
    // Pre-allocate input tensor
    input.resize(batch_size);
    for (std::vector<std::vector<float>> &batch : input) {
        batch.resize(seq_len);
        for (std::vector<float> &seq : batch) {
            seq.resize(input_size, 0.0f);
        }
    }

    // Pre-allocate target and prediction vectors
    target.resize(target_size, 0.0f);
    pred.resize(target_size, 0.0f);

    // Pre-allocate output structure
    output.sequence_output.resize(batch_size);
    for (std::vector<std::vector<float>> &batch : output.sequence_output) {
        batch.resize(seq_len);
        for (std::vector<float> &seq : batch) {
            seq.resize(hidden_size, 0.0f);
        }
    }

    // Pre-allocate hidden and cell states
    output.final_hidden.resize(num_layers);
    output.final_cell.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        output.final_hidden[i].resize(hidden_size, 0.0f);
        output.final_cell[i].resize(hidden_size, 0.0f);
    }
}

void LSTMPredictor::save_model_state(std::ofstream& file) {
    try {
        // Print configuration being saved for debugging
        std::cout << "Saving LSTM model with config: classes=" << num_classes 
                  << ", input=" << input_size
                  << ", hidden=" << hidden_size
                  << ", layers=" << num_layers
                  << ", seq_len=" << seq_length << std::endl;
                  
        // Save model configuration
        file.write(reinterpret_cast<const char*>(&num_classes), sizeof(int));
        file.write(reinterpret_cast<const char*>(&input_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
        file.write(reinterpret_cast<const char*>(&seq_length), sizeof(int));
        file.write(reinterpret_cast<const char*>(&batch_first), sizeof(bool));

        // Save weights
        save_weights(file);

        // Save biases
        save_biases(file);

        // Save velocity terms for momentum
        // Save LSTM layer velocities
        for (int layer = 0; layer < num_layers; ++layer) {
            // Save weight_ih velocity
            size_t ih_rows = velocity_weight_ih[layer].size();
            size_t ih_cols = velocity_weight_ih[layer][0].size();
            file.write(reinterpret_cast<const char*>(&ih_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&ih_cols), sizeof(size_t));
            for (const std::vector<float> &row : velocity_weight_ih[layer]) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }

            // Save weight_hh velocity
            size_t hh_rows = velocity_weight_hh[layer].size();
            size_t hh_cols = velocity_weight_hh[layer][0].size();
            file.write(reinterpret_cast<const char*>(&hh_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&hh_cols), sizeof(size_t));
            for (const std::vector<float> &row : velocity_weight_hh[layer]) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }

            // Save bias velocities
            file.write(reinterpret_cast<const char*>(velocity_bias_ih[layer].data()), 
                      velocity_bias_ih[layer].size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(velocity_bias_hh[layer].data()), 
                      velocity_bias_hh[layer].size() * sizeof(float));
        }

        // Save FC layer velocities
        size_t fc_rows = velocity_fc_weight.size();
        size_t fc_cols = velocity_fc_weight[0].size();
        file.write(reinterpret_cast<const char*>(&fc_rows), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&fc_cols), sizeof(size_t));
        for (const std::vector<float> &row : velocity_fc_weight) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        file.write(reinterpret_cast<const char*>(velocity_fc_bias.data()), 
                  velocity_fc_bias.size() * sizeof(float));
                  
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving model to stream: " + std::string(e.what()));
    }
}

void LSTMPredictor::load_model_state(std::ifstream& file) {
    try {
        int loaded_num_classes;
        int loaded_input_size;
        int loaded_hidden_size;
        int loaded_num_layers;
        int loaded_seq_length;
        bool loaded_batch_first;

        file.read(reinterpret_cast<char*>(&loaded_num_classes), sizeof(int));
        file.read(reinterpret_cast<char*>(&loaded_input_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&loaded_hidden_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&loaded_num_layers), sizeof(int));
        file.read(reinterpret_cast<char*>(&loaded_seq_length), sizeof(int));
        file.read(reinterpret_cast<char*>(&loaded_batch_first), sizeof(bool));
        
        // Print loaded and current config for debugging
        std::cout << "LSTM model dimensions: classes=" << loaded_num_classes 
                  << ", input=" << loaded_input_size
                  << ", hidden=" << loaded_hidden_size
                  << ", layers=" << loaded_num_layers
                  << ", seq_len=" << loaded_seq_length << std::endl;
        
        // Load weights and biases
        load_weights(file);
        load_biases(file);

        // Load velocity terms for momentum
        // Load LSTM layer velocities
        for (int layer = 0; layer < num_layers; ++layer) {
            // Load weight_ih velocity
            size_t ih_rows, ih_cols;
            file.read(reinterpret_cast<char*>(&ih_rows), sizeof(size_t));
            file.read(reinterpret_cast<char*>(&ih_cols), sizeof(size_t));
            velocity_weight_ih[layer].resize(ih_rows, std::vector<float>(ih_cols));
            for (std::vector<float> &row : velocity_weight_ih[layer]) {
                file.read(reinterpret_cast<char*>(row.data()), ih_cols * sizeof(float));
            }

            // Load weight_hh velocity
            size_t hh_rows, hh_cols;
            file.read(reinterpret_cast<char*>(&hh_rows), sizeof(size_t));
            file.read(reinterpret_cast<char*>(&hh_cols), sizeof(size_t));
            velocity_weight_hh[layer].resize(hh_rows, std::vector<float>(hh_cols));
            for (std::vector<float> &row : velocity_weight_hh[layer]) {
                file.read(reinterpret_cast<char*>(row.data()), hh_cols * sizeof(float));
            }

            // Load bias velocities
            velocity_bias_ih[layer].resize(4 * hidden_size);
            velocity_bias_hh[layer].resize(4 * hidden_size);
            file.read(reinterpret_cast<char*>(velocity_bias_ih[layer].data()), 
                     4 * hidden_size * sizeof(float));
            file.read(reinterpret_cast<char*>(velocity_bias_hh[layer].data()), 
                     4 * hidden_size * sizeof(float));
        }

        // Load FC layer velocities
        size_t fc_rows, fc_cols;
        file.read(reinterpret_cast<char*>(&fc_rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&fc_cols), sizeof(size_t));
        velocity_fc_weight.resize(fc_rows, std::vector<float>(fc_cols));
        for (std::vector<float> &row : velocity_fc_weight) {
            file.read(reinterpret_cast<char*>(row.data()), fc_cols * sizeof(float));
        }
        velocity_fc_bias.resize(num_classes);
        file.read(reinterpret_cast<char*>(velocity_fc_bias.data()), 
                 num_classes * sizeof(float));
                 
        clear_update_state();
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading model from stream: " + std::string(e.what()));
    }
}