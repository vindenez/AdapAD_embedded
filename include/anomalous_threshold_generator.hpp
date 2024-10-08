#ifndef ANOMALOUS_THRESHOLD_GENERATOR_HPP
#define ANOMALOUS_THRESHOLD_GENERATOR_HPP

#include "lstm_predictor.hpp"
#include <vector>

class AnomalousThresholdGenerator {
public:
    // Constructor to initialize using weights and biases
    AnomalousThresholdGenerator(const std::vector<std::vector<float>>& weight_ih,
                                const std::vector<std::vector<float>>& weight_hh,
                                const std::vector<float>& bias_ih,
                                const std::vector<float>& bias_hh);

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
