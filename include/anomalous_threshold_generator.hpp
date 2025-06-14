#ifndef ANOMALOUS_THRESHOLD_GENERATOR_HPP
#define ANOMALOUS_THRESHOLD_GENERATOR_HPP

#include "lstm_predictor_factory.hpp"
#include <fstream>
#include <memory>
#include <vector>

class AnomalousThresholdGenerator {
  public:
    AnomalousThresholdGenerator(int lstm_layer, int lstm_unit, int lookback_len,
                                int prediction_len);

    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
    train(int epoch, float lr, const std::vector<float> &data2learn, int train_size);

    float generate(const std::vector<float> &prediction_errors, float minimal_threshold);

    void update(int epoch_update, float lr_update, const std::vector<float> &past_errors,
                float recent_error);

    std::vector<float> forward(const std::vector<std::vector<std::vector<float>>> &x) {
        return generator->forward(x);
    }

    void eval() { generator->eval(); }
    void train() { generator->train(); }

    void reset_states();

    // Model save/load methods
    void save_weights(std::ofstream &file);
    void save_biases(std::ofstream &file);
    void load_weights(std::ifstream &file);
    void load_biases(std::ifstream &file);
    void save_model_state(std::ofstream &file);
    void load_model_state(std::ifstream &file);
    void initialize_layer_cache();
    bool is_layer_cache_initialized() const { return generator->is_layer_cache_initialized(); }
    void clear_update_state();

    bool is_training() const { return generator ? generator->is_training() : false; }

    bool is_online_learning() const { return generator ? generator->is_online_learning() : false; }

    void learn() {
        if (generator) {
            generator->learn();
        }
    }

  private:
    std::unique_ptr<LSTMPredictor> generator;

    // Pre-allocated vectors for update
    std::vector<std::vector<std::vector<float>>> update_input;
    std::vector<float> update_target;
    std::vector<float> update_pred;
    LSTMPredictor::LSTMOutput update_output;

    std::pair<std::vector<std::vector<float>>, std::vector<float>> create_sliding_windows(const std::vector<float> &data, int lookback_len, int prediction_len);

    int lookback_len;
    int prediction_len;
};

#endif // ANOMALOUS_THRESHOLD_GENERATOR_HPP
