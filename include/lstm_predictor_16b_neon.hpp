#pragma once
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <arm_fp16.h>  
#include <arm_neon.h>

class LSTMPredictor16bNEON {
public:
    struct LSTMLayer16bit {
        // 16-bit weights [i,f,g,o] gates are stacked
        std::vector<std::vector<float16_t>> weight_ih;  // (4*hidden_size, input_size)
        std::vector<std::vector<float16_t>> weight_hh;  // (4*hidden_size, hidden_size)
        std::vector<float16_t> bias_ih;                 // (4*hidden_size)
        std::vector<float16_t> bias_hh;                 // (4*hidden_size)
    };

    struct LSTMGradients16bit {
        // All gradients also in fp16 for speed
        std::vector<std::vector<float16_t>> weight_ih_grad;
        std::vector<std::vector<float16_t>> weight_hh_grad;
        std::vector<float16_t> bias_ih_grad;
        std::vector<float16_t> bias_hh_grad;
    };

    struct LSTMOutput16bit {
        std::vector<std::vector<std::vector<float16_t>>> sequence_output;   // [batch_size][seq_len][hidden_size]
        std::vector<std::vector<float16_t>> final_hidden;                   // [num_layers][hidden_size]
        std::vector<std::vector<float16_t>> final_cell;                     // [num_layers][hidden_size]
    };

    // Constructor and methods
    LSTMPredictor16bNEON(int num_classes, int input_size, int hidden_size, 
                      int num_layers, int lookback_len, 
                      bool batch_first = true);

    virtual LSTMOutput16bit forward(const std::vector<std::vector<std::vector<float16_t>>>& x,
                    const std::vector<std::vector<float16_t>>* initial_hidden = nullptr,
                    const std::vector<std::vector<float16_t>>* initial_cell = nullptr);

    virtual std::vector<float16_t> get_final_prediction(const LSTMOutput16bit& lstm_output);

    // Training methods
    virtual void train_step(const std::vector<std::vector<std::vector<float16_t>>>& x,
                   const std::vector<float16_t>& target,
                   const LSTMOutput16bit& lstm_output,
                   float16_t learning_rate);
    
    virtual std::vector<float16_t> compute_mse_loss_gradient(
        const std::vector<float16_t>& output,
        const std::vector<float16_t>& target);

    void set_random_seed(unsigned seed) {
        random_seed = seed;
        initialize_weights();
    }
    
    // Weight setters for loading pretrained models (converted from fp32)
    void set_lstm_weights(int layer, const std::vector<std::vector<float>>& w_ih,
                         const std::vector<std::vector<float>>& w_hh);
    void set_lstm_bias(int layer, const std::vector<float>& b_ih,
                      const std::vector<float>& b_hh);
    void set_fc_weights(const std::vector<std::vector<float>>& weights,
                       const std::vector<float>& bias);

    void reset_states();

    std::vector<LSTMLayer16bit> get_weights() const {
        return lstm_layers;
    }
    
    void set_weights(const std::vector<LSTMLayer16bit>& weights);
    
    std::vector<LSTMGradients16bit> get_last_gradients() const {
        return last_gradients;
    }

    int get_num_layers() const { return num_layers; }

    void eval() { 
        training_mode = false; 
        online_learning_mode = false;
    }
    
    void train() { 
        training_mode = true; 
    }

    void learn() { 
        training_mode = true; 
        online_learning_mode = true;
    }

    bool is_training() const { return training_mode; }
    bool is_online_learning() const { return online_learning_mode; }

    void set_is_cache_initialized(bool value) { is_cache_initialized = value; }
    
    bool is_layer_cache_initialized() const { return is_cache_initialized; }

    // Model save/load methods
    void save_weights(std::ofstream& file);
    void save_biases(std::ofstream& file);
    void load_weights(std::ifstream& file);
    void load_biases(std::ifstream& file);
    void save_layer_cache(std::ofstream& file) const;
    void load_layer_cache(std::ifstream& file);
    void initialize_layer_cache();

    std::pair<std::vector<float16_t>, std::vector<float16_t>> get_state() const {
        return {h_state[0], c_state[0]};  // Return first layer's states
    }
    
    void set_state(const std::pair<std::vector<float16_t>, std::vector<float16_t>>& state) {
        if (h_state.size() > 0 && c_state.size() > 0) {
            h_state[0] = state.first;
            c_state[0] = state.second;
        }
    }

    void clear_update_state();

protected:
    unsigned random_seed;
    // Model dimensions
    int num_classes;
    int num_layers;
    int input_size;
    int hidden_size;
    int seq_length;
    bool batch_first;

    std::vector<LSTMLayer16bit> lstm_layers;

    // Final linear layer weights
    std::vector<std::vector<float16_t>> fc_weight;
    std::vector<float16_t> fc_bias;

    // Hidden states
    std::vector<std::vector<float16_t>> h_state; // [num_layers][hidden_size]
    std::vector<std::vector<float16_t>> c_state; // [num_layers][hidden_size]

    struct LSTMCacheEntry16bit {
        std::vector<float16_t> input;
        std::vector<float16_t> prev_hidden;
        std::vector<float16_t> prev_cell;
        std::vector<float16_t> cell_state;
        std::vector<float16_t> input_gate;
        std::vector<float16_t> forget_gate;
        std::vector<float16_t> cell_candidate;
        std::vector<float16_t> output_gate;
        std::vector<float16_t> hidden_state;
        
        LSTMCacheEntry16bit() = default;
        LSTMCacheEntry16bit(const LSTMCacheEntry16bit& other) = default;
        LSTMCacheEntry16bit& operator=(const LSTMCacheEntry16bit& other) = default;
    };
    std::vector<std::vector<std::vector<LSTMCacheEntry16bit>>> layer_cache;

    // Store last gradients for testing
    std::vector<LSTMGradients16bit> last_gradients;

    // Helper functions for 16-bit operations using NEON
    static float16x8_t sigmoid_neon_fp16(float16x8_t x);
    static float16x8_t tanh_neon_fp16(float16x8_t x);
    static float16x4_t sigmoid_neon_fp16_half(float16x4_t x);
    static float16x4_t tanh_neon_fp16_half(float16x4_t x);
    
    virtual std::vector<float16_t> lstm_cell_forward(const std::vector<float16_t>& input,
                                                    std::vector<float16_t>& h_state,
                                                    std::vector<float16_t>& c_state,
                                                    const LSTMLayer16bit& layer);
    
    // Training helper functions - all in fp16
    virtual void backward_linear_layer(const std::vector<float16_t>& grad_output,
                             const std::vector<float16_t>& last_hidden,
                             std::vector<std::vector<float16_t>>& weight_grad,
                             std::vector<float16_t>& bias_grad,
                             std::vector<float16_t>& input_grad);
    
    virtual std::vector<LSTMGradients16bit> backward_lstm_layer(const std::vector<float16_t>& grad_output,
                                                const std::vector<std::vector<std::vector<LSTMCacheEntry16bit>>>& cache,
                                                float16_t learning_rate);

    int current_layer = 0;
    size_t current_batch{0};  
    size_t current_timestep{0};

    void initialize_weights();

    virtual void apply_sgd_update(std::vector<std::vector<float16_t>>& weights,
                        std::vector<std::vector<float16_t>>& grads,
                        float16_t learning_rate,
                        float16_t momentum = 0.9f16);
    
    virtual void apply_sgd_update(std::vector<float16_t>& biases,
                        std::vector<float16_t>& grads,
                        float16_t learning_rate,
                        float16_t momentum = 0.9f16);

    bool training_mode = true;
    bool online_learning_mode = false;

    bool is_cache_initialized = false;

    size_t current_cache_size = 0;  
    
    // Momentum velocity terms (also in fp16)
    std::vector<std::vector<std::vector<float16_t>>> velocity_weight_ih;  // [num_layers][4*hidden_size][input_size]
    std::vector<std::vector<std::vector<float16_t>>> velocity_weight_hh;  // [num_layers][4*hidden_size][hidden_size]
    std::vector<std::vector<float16_t>> velocity_bias_ih;                 // [num_layers][4*hidden_size]
    std::vector<std::vector<float16_t>> velocity_bias_hh;                 // [num_layers][4*hidden_size]
    std::vector<std::vector<float16_t>> velocity_fc_weight;              // [num_classes][hidden_size]
    std::vector<float16_t> velocity_fc_bias;                             // [num_classes]
};