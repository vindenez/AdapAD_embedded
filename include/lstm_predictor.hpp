#ifndef LSTM_PREDICTOR_HPP
#define LSTM_PREDICTOR_HPP

#include <vector>
#include <tuple>

class LSTMPredictor {
public:
    // Constructor to initialize with input size, hidden size, and other hyperparameters
    LSTMPredictor(int input_size, int hidden_size, int num_layers, int lookback_len);
    
    // Constructor to initialize with weights and biases for each gate
    LSTMPredictor(const std::vector<std::vector<float>>& weight_ih_input,
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
                  const std::vector<float>& bias_hh_cell);

    // Simplified constructor for fewer parameters (matching NormalDataPredictor)
    LSTMPredictor(const std::vector<std::vector<float>>& weight_ih,
                  const std::vector<std::vector<float>>& weight_hh,
                  const std::vector<float>& bias_ih,
                  const std::vector<float>& bias_hh);

    // Updated forward method
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& input,
        const std::vector<float>& prev_h,
        const std::vector<float>& prev_c);

    int get_input_size() const;
    int get_hidden_size() const;

    // Copy constructor
    LSTMPredictor(const LSTMPredictor& other);
    
    // Getter methods for weights and biases
    const std::vector<std::vector<float>>& get_weight_ih_input() const { return weight_ih_input; }
    const std::vector<std::vector<float>>& get_weight_hh_input() const { return weight_hh_input; }
    const std::vector<float>& get_bias_ih_input() const { return bias_ih_input; }
    const std::vector<float>& get_bias_hh_input() const { return bias_hh_input; }

    // Setter methods for weights and biases
    void set_weight_ih_input(const std::vector<std::vector<float>>& w) { weight_ih_input = w; }
    void set_weight_hh_input(const std::vector<std::vector<float>>& w) { weight_hh_input = w; }
    void set_bias_ih_input(const std::vector<float>& b) { bias_ih_input = b; }
    void set_bias_hh_input(const std::vector<float>& b) { bias_hh_input = b; }

    // Update parameters method
    void update_parameters(const std::vector<std::vector<float>>& dw_ih, 
                           const std::vector<std::vector<float>>& dw_hh,
                           const std::vector<float>& db_ih, 
                           const std::vector<float>& db_hh, 
                           float learning_rate);

private:
    int input_size;
    int hidden_size;
    int num_layers;
    int lookback_len;

    // Weights and biases for each gate
    std::vector<std::vector<float>> weight_ih_input;
    std::vector<std::vector<float>> weight_hh_input;
    std::vector<float> bias_ih_input;
    std::vector<float> bias_hh_input;

    std::vector<std::vector<float>> weight_ih_forget;
    std::vector<std::vector<float>> weight_hh_forget;
    std::vector<float> bias_ih_forget;
    std::vector<float> bias_hh_forget;

    std::vector<std::vector<float>> weight_ih_output;
    std::vector<std::vector<float>> weight_hh_output;
    std::vector<float> bias_ih_output;
    std::vector<float> bias_hh_output;

    std::vector<std::vector<float>> weight_ih_cell;
    std::vector<std::vector<float>> weight_hh_cell;
    std::vector<float> bias_ih_cell;
    std::vector<float> bias_hh_cell;

    std::vector<float> h; // hidden state
    std::vector<float> c; // cell state
};

#endif // LSTM_PREDICTOR_HPP