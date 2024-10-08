#ifndef LSTM_PREDICTOR_HPP
#define LSTM_PREDICTOR_HPP

#include <vector>

class LSTMPredictor {
public:
    // Constructor to initialize with input size, hidden size, and other hyperparameters
    LSTMPredictor(int input_size, int hidden_size, int num_layers, int lookback_len);
    
    // Constructor to initialize with weights and biases
    LSTMPredictor(const std::vector<std::vector<float>>& weight_ih,
                  const std::vector<std::vector<float>>& weight_hh,
                  const std::vector<float>& bias_ih,
                  const std::vector<float>& bias_hh);

    std::vector<float> forward(const std::vector<float>& input);

private:
    int input_size;
    int hidden_size;
    int num_layers;
    int lookback_len;

    std::vector<std::vector<float>> weight_ih;
    std::vector<std::vector<float>> weight_hh;
    std::vector<float> bias_ih;
    std::vector<float> bias_hh;

    std::vector<float> h; // hidden state
    std::vector<float> c; // cell state
};

#endif // LSTM_PREDICTOR_HPP
