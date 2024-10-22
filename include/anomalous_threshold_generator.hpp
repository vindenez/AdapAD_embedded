#ifndef ANOMALOUS_THRESHOLD_GENERATOR_HPP
#define ANOMALOUS_THRESHOLD_GENERATOR_HPP

#include "lstm_predictor.hpp"
#include <vector>

class AnomalousThresholdGenerator {
public:
    AnomalousThresholdGenerator(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len);

    void train(int num_epochs, float learning_rate, const std::vector<float>& data_to_learn);
    float update(int num_epochs, float lr_update, const std::vector<float>& past_errors, float recent_error);
    float generate(const std::vector<float>& prediction_errors, float minimal_threshold);
    void train();
    void eval();

private:
    int lookback_len;
    int prediction_len;
    LSTMPredictor generator;
    bool is_training;

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    sliding_windows(const std::vector<float>& data, int window_size, int prediction_len);

    float compute_mse_loss(const std::vector<float>& output, const std::vector<float>& target);
    std::vector<float> compute_mse_loss_gradient(const std::vector<float>& output, const std::vector<float>& target);
};

#endif // ANOMALOUS_THRESHOLD_GENERATOR_HPP
