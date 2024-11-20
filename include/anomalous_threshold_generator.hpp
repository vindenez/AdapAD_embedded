#ifndef ANOMALOUS_THRESHOLD_GENERATOR_HPP
#define ANOMALOUS_THRESHOLD_GENERATOR_HPP

#include "lstm_predictor.hpp"
#include <vector>
#include <memory>

class AnomalousThresholdGenerator {
public:
    AnomalousThresholdGenerator(int lstm_layer, int lstm_unit, 
                               int lookback_len, int prediction_len);
    
    // Train the generator on error data - match NormalDataPredictor's interface
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
    train(int epoch, float lr, const std::vector<float>& data2learn);
    
    // Make a single prediction (renamed from generate)
    float generate(const std::vector<float>& prediction_errors, float minimal_threshold);
    
    // Update the model with new error observations
    void update(int epoch_update, float lr_update,
                const std::vector<float>& past_errors, float recent_error);
    
    // Add these new methods to delegate to LSTM predictor
    void eval() { generator->eval(); }
    void train() { generator->train(); }
    LSTMPredictor::LSTMOutput forward(const std::vector<std::vector<std::vector<float>>>& x) {
        return generator->forward(x);
    }
    std::vector<float> get_final_prediction(const LSTMPredictor::LSTMOutput& output) {
        return generator->get_final_prediction(output);
    }
    
    // Add these delegate methods
    void reset_states() { generator->reset_states(); }
    void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                   const std::vector<float>& target,
                   float learning_rate) {
        generator->train_step(x, target, learning_rate);
    }

private:
    int lookback_len;
    int prediction_len;
    std::unique_ptr<LSTMPredictor> generator;
    
    std::pair<std::vector<std::vector<float>>, std::vector<float>>
    create_sliding_windows(const std::vector<float>& data);
};

#endif // ANOMALOUS_THRESHOLD_GENERATOR_HPP