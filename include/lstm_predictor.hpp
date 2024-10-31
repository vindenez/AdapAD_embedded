#ifndef LSTM_PREDICTOR_HPP
#define LSTM_PREDICTOR_HPP

#include <vector>
#include <tuple>
#include <string>

class LSTMPredictor {
public:
    // Constructors
    LSTMPredictor(int input_size, int hidden_size, int output_size, int num_layers, int lookback_len);

    LSTMPredictor(
        const std::vector<std::vector<float>>& weight_ih_input,
        const std::vector<std::vector<float>>& weight_hh_input,
        const std::vector<float>& bias_ih_input,
        const std::vector<float>& bias_hh_input,
        const std::vector<std::vector<float>>& weight_ih_forget,
        const std::vector<std::vector<float>>& weight_hh_forget,
        const std::vector<float>& bias_ih_forget,
        const std::vector<float>& bias_hh_forget,
        const std::vector<std::vector<float>>& weight_ih_output,
        const std::vector<std::vector<float>>& weight_hh_output,
        const std::vector<float>& bias_ih_output,
        const std::vector<float>& bias_hh_output,
        const std::vector<std::vector<float>>& weight_ih_cell,
        const std::vector<std::vector<float>>& weight_hh_cell,
        const std::vector<float>& bias_ih_cell,
        const std::vector<float>& bias_hh_cell,
        int input_size,
        int hidden_size
    );

    // Copy constructor
    LSTMPredictor(const LSTMPredictor& other);

    // Forward pass methods
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& input,
        const std::vector<float>& prev_h,
        const std::vector<float>& prev_c
    );

    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
               std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
    forward_step(
        const std::vector<float>& input_vec,
        const std::vector<float>& prev_h,
        const std::vector<float>& prev_c
    );

    std::vector<float> forward(const std::vector<std::vector<std::vector<float>>>& input_sequence);
    
    // Training methods
    void backward(const std::vector<float>& targets, const std::string& loss_function = "MSE");
    void zero_grad();
    void update_parameters(float learning_rate);
    void init_adam_optimizer(float learning_rate);
    void update_parameters_adam(float learning_rate);
    void train();
    void eval();

    // Getters
    int get_input_size() const;
    int get_hidden_size() const;
    const std::vector<float>& get_h() const;
    const std::vector<float>& get_c() const;

    // Getter methods for weights and biases
    // Input gate
    const std::vector<std::vector<float>>& get_weight_ih_input() const { return weight_ih_input; }
    const std::vector<std::vector<float>>& get_weight_hh_input() const { return weight_hh_input; }
    const std::vector<float>& get_bias_ih_input() const { return bias_ih_input; }
    const std::vector<float>& get_bias_hh_input() const { return bias_hh_input; }

    // Forget gate
    const std::vector<std::vector<float>>& get_weight_ih_forget() const { return weight_ih_forget; }
    const std::vector<std::vector<float>>& get_weight_hh_forget() const { return weight_hh_forget; }
    const std::vector<float>& get_bias_ih_forget() const { return bias_ih_forget; }
    const std::vector<float>& get_bias_hh_forget() const { return bias_hh_forget; }

    // Output gate
    const std::vector<std::vector<float>>& get_weight_ih_output() const { return weight_ih_output; }
    const std::vector<std::vector<float>>& get_weight_hh_output() const { return weight_hh_output; }
    const std::vector<float>& get_bias_ih_output() const { return bias_ih_output; }
    const std::vector<float>& get_bias_hh_output() const { return bias_hh_output; }

    // Cell gate
    const std::vector<std::vector<float>>& get_weight_ih_cell() const { return weight_ih_cell; }
    const std::vector<std::vector<float>>& get_weight_hh_cell() const { return weight_hh_cell; }
    const std::vector<float>& get_bias_ih_cell() const { return bias_ih_cell; }
    const std::vector<float>& get_bias_hh_cell() const { return bias_hh_cell; }

    // Setter methods for weights and biases
    // Input gate
    void set_weight_ih_input(const std::vector<std::vector<float>>& w) { weight_ih_input = w; }
    void set_weight_hh_input(const std::vector<std::vector<float>>& w) { weight_hh_input = w; }
    void set_bias_ih_input(const std::vector<float>& b) { bias_ih_input = b; }
    void set_bias_hh_input(const std::vector<float>& b) { bias_hh_input = b; }

    // Forget gate
    void set_weight_ih_forget(const std::vector<std::vector<float>>& w) { weight_ih_forget = w; }
    void set_weight_hh_forget(const std::vector<std::vector<float>>& w) { weight_hh_forget = w; }
    void set_bias_ih_forget(const std::vector<float>& b) { bias_ih_forget = b; }
    void set_bias_hh_forget(const std::vector<float>& b) { bias_hh_forget = b; }

    // Output gate
    void set_weight_ih_output(const std::vector<std::vector<float>>& w) { weight_ih_output = w; }
    void set_weight_hh_output(const std::vector<std::vector<float>>& w) { weight_hh_output = w; }
    void set_bias_ih_output(const std::vector<float>& b) { bias_ih_output = b; }
    void set_bias_hh_output(const std::vector<float>& b) { bias_hh_output = b; }

    // Cell gate
    void set_weight_ih_cell(const std::vector<std::vector<float>>& w) { weight_ih_cell = w; }
    void set_weight_hh_cell(const std::vector<std::vector<float>>& w) { weight_hh_cell = w; }
    void set_bias_ih_cell(const std::vector<float>& b) { bias_ih_cell = b; }
    void set_bias_hh_cell(const std::vector<float>& b) { bias_hh_cell = b; }

private:
    // Dimensions
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;
    int lookback_len;

    float learning_rate;

    // Weights and biases for LSTM gates
    // Input gate
    std::vector<std::vector<float>> weight_ih_input;
    std::vector<std::vector<float>> weight_hh_input;
    std::vector<float> bias_ih_input;
    std::vector<float> bias_hh_input;

    // Forget gate
    std::vector<std::vector<float>> weight_ih_forget;
    std::vector<std::vector<float>> weight_hh_forget;
    std::vector<float> bias_ih_forget;
    std::vector<float> bias_hh_forget;

    // Output gate
    std::vector<std::vector<float>> weight_ih_output;
    std::vector<std::vector<float>> weight_hh_output;
    std::vector<float> bias_ih_output;
    std::vector<float> bias_hh_output;

    // Cell gate
    std::vector<std::vector<float>> weight_ih_cell;
    std::vector<std::vector<float>> weight_hh_cell;
    std::vector<float> bias_ih_cell;
    std::vector<float> bias_hh_cell;

    // Fully connected layer
    std::vector<std::vector<float>> fc_weights;     // Size: output_size x hidden_size
    std::vector<float> fc_bias;                     // Size: output_size

    // Hidden and cell states
    std::vector<float> h;       // Current hidden state
    std::vector<float> c;       // Current cell state
    std::vector<float> h_init;  // Initial hidden state
    std::vector<float> c_init;  // Initial cell state

    // Training state
    bool is_training;

    // Stored activations and inputs for backpropagation
    std::vector<std::vector<float>> x_t_list;
    std::vector<std::vector<float>> i_t_list, f_t_list, o_t_list, g_t_list;
    std::vector<std::vector<float>> c_t_list, h_t_list;
    std::vector<std::vector<float>> outputs_list;

    // Gradients for LSTM gates
    // Input gate
    std::vector<std::vector<float>> dw_ih_input, dw_hh_input;
    std::vector<float> db_ih_input, db_hh_input;

    // Forget gate
    std::vector<std::vector<float>> dw_ih_forget, dw_hh_forget;
    std::vector<float> db_ih_forget, db_hh_forget;

    // Output gate
    std::vector<std::vector<float>> dw_ih_output, dw_hh_output;
    std::vector<float> db_ih_output, db_hh_output;

    // Cell gate
    std::vector<std::vector<float>> dw_ih_cell, dw_hh_cell;
    std::vector<float> db_ih_cell, db_hh_cell;

    // Gradients for fully connected layer
    std::vector<std::vector<float>> dw_fc_weights;
    std::vector<float> db_fc_bias;

    // Adam optimizer parameters
    float beta1;
    float beta2;
    float epsilon;
    int t;  // Time step

    // First moment estimates (m_) for weights and biases
    // Input gate
    std::vector<std::vector<float>> m_w_ih_input, m_w_hh_input;
    std::vector<float> m_b_ih_input, m_b_hh_input;

    // Forget gate
    std::vector<std::vector<float>> m_w_ih_forget, m_w_hh_forget;
    std::vector<float> m_b_ih_forget, m_b_hh_forget;

    // Output gate
    std::vector<std::vector<float>> m_w_ih_output, m_w_hh_output;
    std::vector<float> m_b_ih_output, m_b_hh_output;

    // Cell gate
    std::vector<std::vector<float>> m_w_ih_cell, m_w_hh_cell;
    std::vector<float> m_b_ih_cell, m_b_hh_cell;

    // Fully connected layer
    std::vector<std::vector<float>> m_fc_weights;
    std::vector<float> m_fc_bias;

    // Second moment estimates (v_) for weights and biases
    // Input gate
    std::vector<std::vector<float>> v_w_ih_input, v_w_hh_input;
    std::vector<float> v_b_ih_input, v_b_hh_input;

    // Forget gate
    std::vector<std::vector<float>> v_w_ih_forget, v_w_hh_forget;
    std::vector<float> v_b_ih_forget, v_b_hh_forget;

    // Output gate
    std::vector<std::vector<float>> v_w_ih_output, v_w_hh_output;
    std::vector<float> v_b_ih_output, v_b_hh_output;

    // Cell gate
    std::vector<std::vector<float>> v_w_ih_cell, v_w_hh_cell;
    std::vector<float> v_b_ih_cell, v_b_hh_cell;

    // Fully connected layer
    std::vector<std::vector<float>> v_fc_weights;
    std::vector<float> v_fc_bias;
};

#endif // LSTM_PREDICTOR_HPP
