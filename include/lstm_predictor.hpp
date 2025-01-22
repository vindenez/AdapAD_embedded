#pragma once
#include <vector>
#include <cmath>
#include <tuple>
#include <random>

class LSTMPredictor {
public:
    // Define all structs first
    struct LSTMLayer {
        // Change weight matrix organization to match PyTorch
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
    
    void set_random_seed(unsigned seed) {
        random_seed = seed;
        initialize_weights();
    }

    ~LSTMPredictor();
    
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
                   float learning_rate);
    
    float compute_loss(const std::vector<float>& output,
                      const std::vector<float>& target);

    std::vector<float> get_final_prediction(const LSTMOutput& lstm_output);

    // Add these testing methods
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

    // Add these methods for gradient checking
    std::vector<LSTMLayer> get_weights() const {
        return lstm_layers;
    }
    
    void set_weights(const std::vector<LSTMLayer>& weights);
    
    std::vector<LSTMGradients> get_last_gradients() const {
        return last_gradients;
    }

    int get_num_layers() const { return num_layers; }

    void eval() { training_mode = false; }
    void train() { training_mode = true; }
    bool is_training() const { return training_mode; }

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
        
        // Add constructor to properly initialize vectors
        LSTMCacheEntry() = default;
        
        // Add copy constructor to ensure proper deep copying
        LSTMCacheEntry(const LSTMCacheEntry& other) = default;
        
        // Add assignment operator
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

    // Adam optimizer state variables
    bool adam_initialized = false;
    
    // For LSTM layers
    std::vector<std::vector<std::vector<float>>> m_weight_ih;
    std::vector<std::vector<std::vector<float>>> v_weight_ih;
    std::vector<std::vector<std::vector<float>>> m_weight_hh;
    std::vector<std::vector<std::vector<float>>> v_weight_hh;
    std::vector<std::vector<float>> m_bias_ih;
    std::vector<std::vector<float>> v_bias_ih;
    std::vector<std::vector<float>> m_bias_hh;
    std::vector<std::vector<float>> v_bias_hh;

    // For FC layer
    std::vector<std::vector<float>> m_fc_weight;
    std::vector<std::vector<float>> v_fc_weight;
    std::vector<float> m_fc_bias;
    std::vector<float> v_fc_bias;

    // Adam initialization methods
    void initialize_adam_states();
    bool are_adam_states_initialized() const;

    void apply_adam_update(std::vector<std::vector<float>>& weights, 
                        std::vector<std::vector<float>>& grads,
                        std::vector<std::vector<float>>& m_t, 
                        std::vector<std::vector<float>>& v_t,
                        float learning_rate, float beta1, float beta2, float epsilon, int t);
    
    void apply_adam_update(std::vector<float>& biases, 
                        std::vector<float>& grads,
                        std::vector<float>& m_t, 
                        std::vector<float>& v_t,
                        float learning_rate, float beta1, float beta2, float epsilon, int t);

    bool training_mode = true;  // Add this member variable

};