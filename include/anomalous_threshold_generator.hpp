#ifndef ANOMALOUS_THRESHOLD_GENERATOR_HPP
#define ANOMALOUS_THRESHOLD_GENERATOR_HPP

#include "lstm_predictor.hpp"
#include <vector>
#include <memory>
#include "lstm_predictor.hpp"
#include <fstream>

class AnomalousThresholdGenerator {
public:
    AnomalousThresholdGenerator(int lstm_layer, int lstm_unit, 
                               int lookback_len, int prediction_len);
    
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
    train(int epoch, float lr, const std::vector<float>& data2learn);
    
    // Make a single prediction
    float generate(const std::vector<float>& prediction_errors, float minimal_threshold);
    
    void update(int epoch_update, float lr_update,
                const std::vector<float>& past_errors, float recent_error);
    
    void eval() { generator->eval(); }
    void train() { generator->train(); }
    LSTMPredictor::LSTMOutput forward(const std::vector<std::vector<std::vector<float>>>& x) {
        return generator->forward(x);
    }
    std::vector<float> get_final_prediction(const LSTMPredictor::LSTMOutput& output) {
        return generator->get_final_prediction(output);
    }
    
    void reset_states() { generator->reset_states(); }
    void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                   const std::vector<float>& target,
                   float learning_rate) {
        generator->train_step(x, target, learning_rate);
    }

    // Model save/load methods
    void save_weights(std::ofstream& file);
    void save_biases(std::ofstream& file);
    void load_weights(std::ifstream& file);
    void load_biases(std::ifstream& file);
    void save_layer_cache(std::ofstream& file) const;
    void load_layer_cache(std::ifstream& file);
    void initialize_layer_cache();

private:
    int lookback_len;
    int prediction_len;
    std::unique_ptr<LSTMPredictor> generator;
    
    std::pair<std::vector<std::vector<float>>, std::vector<float>>
    create_sliding_windows(const std::vector<float>& data);
};
#endif // ANOMALOUS_THRESHOLD_GENERATOR_HPP
