#pragma once
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
#include "optimizer.hpp"
#include <memory>

class LSTMPredictor {
public:
    struct LSTMLayer {
        // [i,f,g,o] gates stacked vertically: (4*hidden_size, input_size)
        std::vector<std::vector<float>> weight_ih;  // (4*hidden_size, input_size)
        std::vector<std::vector<float>> weight_hh;  // (4*hidden_size, hidden_size)
        std::vector<float> bias_ih;                 // (4*hidden_size)
        std::vector<float> bias_hh;                 // (4*hidden_size)
    };

    struct LSTMGradients {
        std::vector<std::vector<float>> weight_ih_grad;
        std::vector<std::vector<float>> weight_hh_grad;
        std::vector<float> bias_ih_grad;
        std::vector<float> bias_hh_grad;
    };

    struct LSTMOutput {
        std::vector<std::vector<std::vector<float>>> sequence_output; // [batch_size][seq_len][hidden_size]
        std::vector<std::vector<float>> final_hidden;  // [num_layers][hidden_size]
        std::vector<std::vector<float>> final_cell;    // [num_layers][hidden_size]
    };

    // Constructor and methods
    LSTMPredictor(int num_classes, int input_size, int hidden_size, 
                  int num_layers, int lookback_len, 
                  bool batch_first = true);

    // Add destructor to ensure proper cleanup
    ~LSTMPredictor();
    
    // Delete copy constructor and assignment operator
    LSTMPredictor(const LSTMPredictor&) = delete;
    LSTMPredictor& operator=(const LSTMPredictor&) = delete;
    
    // Allow move operations
    LSTMPredictor(LSTMPredictor&&) = default;
    LSTMPredictor& operator=(LSTMPredictor&&) = default;
    
    void set_random_seed(unsigned seed) {
        random_seed = seed;
        initialize_weights();
    }
    
    LSTMOutput forward(const std::vector<std::vector<std::vector<float>>>& x,
                      const std::vector<std::vector<float>>* initial_hidden = nullptr,
                      const std::vector<std::vector<float>>* initial_cell = nullptr);
    
    // Weight setters for loading pretrained models
    void set_lstm_weights(int layer, const std::vector<std::vector<float>>& w_ih,
                         const std::vector<std::vector<float>>& w_hh);
    void set_lstm_bias(int layer, const std::vector<float>& b_ih,
                      const std::vector<float>& b_hh);
    void set_fc_weights(const std::vector<std::vector<float>>& weights,
                       const std::vector<float>& bias);

    void reset_states();

    // Training methods
    void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                   const std::vector<float>& target,
                   const LSTMOutput& lstm_output,
                   float learning_rate);
    
    float compute_loss(const std::vector<float>& output,
                      const std::vector<float>& target);

    std::vector<float> get_final_prediction(const LSTMOutput& lstm_output);

    #ifdef TESTING
    float get_weight(int layer, int gate, int input_idx) const {
        // Convert from gate index to PyTorch's layout [i,f,g,o]
        int offset = gate * hidden_size;
        return lstm_layers[layer].weight_ih[offset][input_idx];
    }

    void set_weight(int layer, int gate, int input_idx, float value) {
        // Convert from gate index to PyTorch's layout [i,f,g,o]
        int offset = gate * hidden_size;
        lstm_layers[layer].weight_ih[offset][input_idx] = value;
    }

    float get_weight_gradient(int layer, int gate, int input_idx) const {
        if (layer < num_layers) {
            // Convert from gate index to PyTorch's layout [i,f,g,o]
            int offset = gate * hidden_size;
            return last_gradients[layer].weight_ih_grad[offset][input_idx];
        }
        return 0.0f;
    }
    #endif

    std::vector<LSTMLayer> get_weights() const {
        return lstm_layers;
    }
    
    void set_weights(const std::vector<LSTMLayer>& weights);
    
    std::vector<LSTMGradients> get_last_gradients() const {
        return last_gradients;
    }

    int get_num_layers() const { return num_layers; }

    void eval() { 
        training_mode = false; 
    }
    
    void train() { 
        training_mode = true; 
    }
    bool is_training() const { return training_mode; }

    // Model save/load methods
    void save_weights(std::ofstream& file);
    void save_biases(std::ofstream& file);
    void load_weights(std::ifstream& file);
    void load_biases(std::ifstream& file);
    void save_layer_cache(std::ofstream& file) const;
    void load_layer_cache(std::ifstream& file);
    void initialize_layer_cache();

    std::pair<std::vector<float>, std::vector<float>> get_state() const {
        return {h_state[0], c_state[0]};  // Return first layer's states
    }
    
    void set_state(const std::pair<std::vector<float>, std::vector<float>>& state) {
        if (h_state.size() > 0 && c_state.size() > 0) {
            h_state[0] = state.first;
            c_state[0] = state.second;
        }
    }

    void clear_training_state();

    void clear_temporary_cache();

    void set_optimizer(std::unique_ptr<Optimizer> opt);

private:
    unsigned random_seed;
    // Model dimensions
    int num_classes;
    int num_layers;
    int input_size;
    int hidden_size;
    int seq_length;
    bool batch_first;

    std::vector<LSTMLayer> lstm_layers;

    // Final linear layer weights
    std::vector<std::vector<float>> fc_weight;
    std::vector<float> fc_bias;

    // Hidden states
    std::vector<std::vector<float>> h_state; // [num_layers][hidden_size]
    std::vector<std::vector<float>> c_state; // [num_layers][hidden_size]

    struct LSTMCacheEntry {
        std::vector<float> input;
        std::vector<float> prev_hidden;
        std::vector<float> prev_cell;
        std::vector<float> cell_state;
        std::vector<float> input_gate;
        std::vector<float> forget_gate;
        std::vector<float> cell_gate;
        std::vector<float> output_gate;
        std::vector<float> hidden_state;
        
        LSTMCacheEntry() = default;
        LSTMCacheEntry(const LSTMCacheEntry& other) = default;
        LSTMCacheEntry& operator=(const LSTMCacheEntry& other) = default;
    };
    std::vector<std::vector<std::vector<LSTMCacheEntry>>> layer_cache;

    // Store last gradients for testing
    std::vector<LSTMGradients> last_gradients;

    // Helper functions
    float sigmoid(float x);
    float tanh_custom(float x);
    std::vector<float> lstm_cell_forward(
        const std::vector<float>& input,
        std::vector<float>& h_state,
        std::vector<float>& c_state,
        const LSTMLayer& layer);
    
    // Training helper functions
    void backward_linear_layer(const std::vector<float>& grad_output,
                             const std::vector<float>& last_hidden,
                             std::vector<std::vector<float>>& weight_grad,
                             std::vector<float>& bias_grad,
                             std::vector<float>& input_grad);
    
    std::vector<LSTMGradients> backward_lstm_layer(
        const std::vector<float>& grad_output,
        const std::vector<std::vector<std::vector<LSTMCacheEntry>>>& cache,
        float learning_rate);

    int current_layer = 0;
    size_t current_batch{0};  // Track current batch being processed
    size_t current_timestep{0};

    void initialize_weights();

    bool training_mode = true;
    size_t current_cache_size = 0;  // Track current cache size

    std::unique_ptr<Optimizer> optimizer;
};