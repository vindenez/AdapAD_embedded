#ifndef LSTM_PREDICTOR_HPP
#define LSTM_PREDICTOR_HPP

#include <vector>
#include <tuple>

class LSTMPredictor {
public:
    // Constructors
    LSTMPredictor(int input_size, int hidden_size, int num_layers, int lookback_len);

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
        int hidden_size);

    // Forward pass through the LSTM layer
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& input,
        const std::vector<float>& prev_h,
        const std::vector<float>& prev_c);

    // **Forward step method for training**
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
               std::vector<float>, std::vector<float>, std::vector<float>>
    forward_step(float input, const std::vector<float>& prev_h, const std::vector<float>& prev_c);

    // Getter methods
    int get_input_size() const;
    int get_hidden_size() const;

    // Copy constructor
    LSTMPredictor(const LSTMPredictor& other);

    // **Getter Methods for Weights and Biases**
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

    // **Setter Methods for Weights and Biases**
    // Input gate
    void set_weight_ih_input(const std::vector<std::vector<float>>& w) { weight_ih_input = w; }
    void set_weight_hh_input(const std::vector<std::vector<float>>& w) { weight_hh_input = w; }
    void set_bias_ih_input(const std::vector<float>& b) { bias_ih_input = b; }
    void set_bias_hh_input(const std::vector<float>& b) { bias_hh_input = b; }

    // Similarly, add setter methods for other gates if needed
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

    void train();
    void eval();
    void zero_grad();
    std::vector<float> forward(const std::vector<float>& input);
    const std::vector<float>& get_h() const;
    const std::vector<float>& get_c() const;

private:
    int input_size;
    int hidden_size;
    int num_layers;
    int lookback_len;

    // Weights and biases for each gate
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

    // Hidden and cell states
    std::vector<float> h; // hidden state
    std::vector<float> c; // cell state

    bool is_training;
    std::vector<std::vector<float>> dw_ih_input, dw_hh_input;
    std::vector<float> db_ih_input, db_hh_input;
};

#endif // LSTM_PREDICTOR_HPP
