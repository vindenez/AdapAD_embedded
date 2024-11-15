#ifndef LSTM_PREDICTOR_HPP
#define LSTM_PREDICTOR_HPP

#include <vector>
#include <tuple>
#include <random>
#include <string>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

class LSTMPredictor {
public:
    // Constructor
    LSTMPredictor(int input_size, int hidden_size, int output_size, int num_layers, int lookback_len);
    
    // Main interface
    std::vector<float> forward(const std::vector<std::vector<std::vector<float>>>& input);
    void backward(const std::vector<float>& targets, const std::string& loss_function);
    void zero_grad();
    void update_parameters_adam();
    void train();
    void eval();
    float compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target);
    void reset_states();
    void reshape_input(const std::vector<float>& input_sequence, std::vector<std::vector<std::vector<float>>>& reshaped);

    // Getters
    int get_input_size() const { return input_size; }
    int get_hidden_size() const { return hidden_size; }
    const std::vector<float>& get_h() const { return h; }
    const std::vector<float>& get_c() const { return c; }

    std::vector<float> get_state() const;
    void load_state(const std::vector<float>& state);

    // Add this with the other public methods
    void init_adam_optimizer();

    void ensure_gradient_storage(int layer, int seq_len) {
        if (layer > 0 && (layer_gradients.size() <= layer - 1 || layer_gradients[layer - 1].size() < seq_len)) {
            layer_gradients[layer - 1].resize(seq_len);
            for (auto& timestep_grad : layer_gradients[layer - 1]) {
                timestep_grad.resize(layer == 1 ? input_size : hidden_size, 0.0f);
            }
        }
    }
    bool check_gradients(const std::vector<std::vector<std::vector<float>>>& input, 
                    const std::vector<float>& target,
                    float epsilon = 1e-7,
                    float threshold = 1e-5);

    ~LSTMPredictor() = default;

private:
    // Dimensions
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;
    int lookback_len;
    
    // Layer parameters
    std::vector<std::vector<std::vector<float>>> w_ih;  // [layer][4*hidden_size][input_size or hidden_size]
    std::vector<std::vector<std::vector<float>>> w_hh;  // [layer][4*hidden_size][hidden_size]
    std::vector<std::vector<float>> b_ih;  // [layer][4*hidden_size]
    std::vector<std::vector<float>> b_hh;  // [layer][4*hidden_size]
    
    // Fully connected layer
    std::vector<std::vector<float>> fc_weights;
    std::vector<float> fc_bias;
    
    // States
    std::vector<std::vector<float>> h_states;  // [layer][hidden_size]
    std::vector<std::vector<float>> c_states;  // [layer][hidden_size]
    std::vector<float> h;  // Current hidden state
    std::vector<float> c;  // Current cell state
    
    // Stored activations for backpropagation
    std::vector<std::vector<std::vector<float>>> layer_gradients;
    std::vector<std::vector<std::vector<float>>> layer_inputs;  // [layer][time][size]
    std::vector<std::vector<std::vector<float>>> layer_h_states;  // [layer][time][hidden_size]
    std::vector<std::vector<std::vector<float>>> layer_c_states;  // [layer][time][hidden_size]
    std::vector<std::vector<std::vector<float>>> layer_gates;  // [layer][time][4*hidden_size]
    
    // Gradients
    std::vector<std::vector<std::vector<float>>> dw_ih;  // [layer][4*hidden_size][input_size or hidden_size]
    std::vector<std::vector<std::vector<float>>> dw_hh;  // [layer][4*hidden_size][hidden_size]
    std::vector<std::vector<float>> db_ih;  // [layer][4*hidden_size]
    std::vector<std::vector<float>> db_hh;  // [layer][4*hidden_size]
    std::vector<std::vector<float>> dw_fc;
    std::vector<float> db_fc;

    // Training parameters
    bool is_training{false};
    float dropout_rate{0.2f};
    float lr{0.0001f};
    float beta1{0.9f};
    float beta2{0.999f};
    float epsilon{1e-8f};
    float weight_decay{0.01f};
    
    // RNG
    std::random_device rd;
    std::mt19937 gen{rd()};

    // Helper methods
    std::vector<float> lstm_layer_forward(const std::vector<float>& x, int layer);
    void lstm_layer_backward(int layer, const std::vector<float>& grad_output);
    std::vector<std::vector<float>> init_weights(int rows, int cols, const std::function<float(float)>& init_func);
    std::vector<float> apply_dropout(const std::vector<float>& input);
    void clip_gradients(std::vector<std::vector<float>>& gradients, float max_norm);

    
    // Adam optimizer
    struct AdamState {
        int64_t step{0};
        std::vector<std::vector<float>> exp_avg;
        std::vector<std::vector<float>> exp_avg_sq;
        
        AdamState() = default;
        explicit AdamState(const std::vector<std::vector<float>>& param_size);
        explicit AdamState(const std::vector<float>& param_size);
    };
    
    std::unordered_map<std::string, AdamState> adam_states;

    // Stored sequence data
    std::vector<std::vector<float>> x_t_list;
    std::vector<std::vector<float>> i_t_list;
    std::vector<std::vector<float>> f_t_list;
    std::vector<std::vector<float>> o_t_list;
    std::vector<std::vector<float>> g_t_list;
    std::vector<std::vector<float>> c_t_list;
    std::vector<std::vector<float>> h_t_list;
    std::vector<std::vector<float>> outputs_list;
    
    // Initial states
    std::vector<float> h_init;
    std::vector<float> c_init;

    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
               std::vector<float>, std::vector<float>, std::vector<float>, 
               std::vector<float>> forward_step(const std::vector<float>& x_t,
                                              const std::vector<float>& prev_h,
                                              const std::vector<float>& prev_c);

    
    std::vector<float> compute_numerical_gradient(
        const std::vector<std::vector<std::vector<float>>>& input,
        const std::vector<float>& target,
        std::vector<std::vector<float>>& param,
        size_t i, size_t j,
        float epsilon);

    float vector_magnitude(const std::vector<float>& vec);
};

#endif // LSTM_PREDICTOR_HPP
