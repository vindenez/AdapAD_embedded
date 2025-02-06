#ifndef NORMAL_DATA_PREDICTOR_HPP
#define NORMAL_DATA_PREDICTOR_HPP

#include "lstm_predictor.hpp"
#include <vector>
#include <memory>
#include "lstm_predictor.hpp"

class NormalDataPredictor {
public:
    NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len);
    
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
    train(int epoch, float lr, const std::vector<float>& data2learn);
    
    float predict(const std::vector<std::vector<std::vector<float>>>& observed);
    
    void update(int epoch_update, float lr_update,
                const std::vector<std::vector<std::vector<float>>>& past_observations,
                const std::vector<float>& recent_observation);

    void reset_states() { predictor->reset_states(); }
    void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                   const std::vector<float>& target,
                   float learning_rate) {
        predictor->train_step(x, target, learning_rate);
    }

    // Existing delegate methods
    void eval() { predictor->eval(); }
    void train() { predictor->train(); }
    LSTMPredictor::LSTMOutput forward(const std::vector<std::vector<std::vector<float>>>& x) {
        return predictor->forward(x);
    }
    std::vector<float> get_final_prediction(const LSTMPredictor::LSTMOutput& output) {
        return predictor->get_final_prediction(output);
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
    std::unique_ptr<LSTMPredictor> predictor;
    
    std::pair<std::vector<std::vector<float>>, std::vector<float>>
    create_sliding_windows(const std::vector<float>& data);
};
#endif // NORMAL_DATA_PREDICTOR_HPP
