#ifndef ANOMALOUS_THRESHOLD_GENERATOR_HPP
#define ANOMALOUS_THRESHOLD_GENERATOR_HPP

#include "lstm_predictor.hpp"
#include <vector>

class AnomalousThresholdGenerator {
public:
    // Constructor to initialize using weights and biases for each gate
    AnomalousThresholdGenerator(
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
        const std::vector<float>& bias_hh_cell);

    // Constructor to initialize using hyperparameters
    AnomalousThresholdGenerator(int lookback_len, int prediction_len, float lower_bound, float upper_bound);

    void train(int num_epochs, float learning_rate, const std::vector<std::vector<float>>& x, const std::vector<std::vector<float>>& y);
    void update(int num_epochs, float learning_rate, const std::vector<float>& past_errors, float recent_error);
    float generate(const std::vector<float>& prediction_errors, float minimal_threshold);

private:
    LSTMPredictor generator;
    int lookback_len;
    int prediction_len;
    float lower_bound;
    float upper_bound;
};

#endif // ANOMALOUS_THRESHOLD_GENERATOR_HPP
