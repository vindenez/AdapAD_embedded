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
    void update_parameters_adam(float learning_rate);
    void train();
    void eval();
    float compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target);

    // Getters
    int get_input_size() const { return input_size; }
    int get_hidden_size() const { return hidden_size; }
    const std::vector<float>& get_h() const { return h; }
    const std::vector<float>& get_c() const { return c; }

    // Add this with the other public methods
    void init_adam_optimizer(float learning_rate);

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
    float lr{0.001f};
    float beta1{0.9f};
    float beta2{0.999f};
    float epsilon{1e-8f};
    float weight_decay{0.0f};
    
    // RNG
    std::random_device rd;
    std::mt19937 gen{rd()};

    // Helper methods
    std::vector<float> lstm_layer_forward(const std::vector<float>& x, int layer);
    void lstm_layer_backward(int layer, const std::vector<float>& grad_output);
    std::vector<std::vector<float>> init_weights(int rows, int cols, const std::function<float(float)>& init_func);
    std::vector<float> apply_dropout(const std::vector<float>& input);
    
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
};

#endif // LSTM_PREDICTOR_HPP
