#include "lstm_predictor.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"

// Constructor to initialize with input size, hidden size, and other hyperparameters
LSTMPredictor::LSTMPredictor(int input_size, int hidden_size, int num_layers, int lookback_len)
    : input_size(input_size),
      hidden_size(hidden_size),
      num_layers(num_layers),
      lookback_len(lookback_len),
      h(hidden_size, 0.0f),
      c(hidden_size, 0.0f) {}

// Constructor to initialize with weights and biases
LSTMPredictor::LSTMPredictor(const std::vector<std::vector<float>>& weight_ih,
                             const std::vector<std::vector<float>>& weight_hh,
                             const std::vector<float>& bias_ih,
                             const std::vector<float>& bias_hh)
    : weight_ih(weight_ih), weight_hh(weight_hh), bias_ih(bias_ih), bias_hh(bias_hh),
      input_size(weight_ih[0].size()), hidden_size(weight_ih.size()),
      h(hidden_size, 0.0f), c(hidden_size, 0.0f) {}

std::vector<float> LSTMPredictor::forward(const std::vector<float>& input) {
    // Input gate
    std::vector<float> i = elementwise_add(matrix_vector_mul(weight_ih, input),
                                           matrix_vector_mul(weight_hh, h));
    i = elementwise_add(i, bias_ih);
    for (float& val : i) val = sigmoid(val);

    // Forget gate
    std::vector<float> f = elementwise_add(matrix_vector_mul(weight_ih, input),
                                           matrix_vector_mul(weight_hh, h));
    f = elementwise_add(f, bias_hh);
    for (float& val : f) val = sigmoid(val);

    // Output gate
    std::vector<float> o = elementwise_add(matrix_vector_mul(weight_ih, input),
                                           matrix_vector_mul(weight_hh, h));
    o = elementwise_add(o, bias_ih);
    for (float& val : o) val = sigmoid(val);

    // Cell state update
    std::vector<float> g = elementwise_add(matrix_vector_mul(weight_ih, input),
                                           matrix_vector_mul(weight_hh, h));
    for (float& val : g) val = tanh_func(val);

    c = elementwise_add(elementwise_mul(f, c), elementwise_mul(i, g));

    // Hidden state update
    std::vector<float> tanh_c(c.size());
    for (size_t j = 0; j < c.size(); ++j) {
        tanh_c[j] = tanh_func(c[j]);
    }

    h = elementwise_mul(o, tanh_c);

    return h;
}
