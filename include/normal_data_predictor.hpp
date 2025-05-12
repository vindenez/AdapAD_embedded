#ifndef NORMAL_DATA_PREDICTOR_HPP
#define NORMAL_DATA_PREDICTOR_HPP

#include "lstm_predictor_factory.hpp"
#include <fstream>
#include <memory>
#include <vector>

class NormalDataPredictor {
  public:
    NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len);

    std::pair<std::vector<std::vector<float>>, std::vector<float>> create_sliding_windows(const std::vector<float> &data, int lookback_len, int prediction_len);

    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
    train(int epoch, float lr, const std::vector<float> &data2learn, int train_size);

    float predict(const std::vector<std::vector<std::vector<float>>> &observed);

    void update(int epoch_update, float lr_update,
                const std::vector<std::vector<std::vector<float>>> &past_observations,
                const std::vector<float> &recent_observation);

    std::vector<float> forward(const std::vector<std::vector<std::vector<float>>> &x) {
        return predictor->forward(x);
    }

    void reset_states();

    // Existing delegate methods
    void eval() { predictor->eval(); }
    void train() { predictor->train(); }

    // Model save/load methods
    void save_weights(std::ofstream &file);
    void save_biases(std::ofstream &file);
    void load_weights(std::ifstream &file);
    void load_biases(std::ifstream &file);
    void save_model_state(std::ofstream& file);
    void load_model_state(std::ifstream& file);
    void initialize_layer_cache();
    void clear_update_state();
    // Add this method to expose training mode status
    bool is_training() const { return predictor ? predictor->is_training() : false; }

    bool is_online_learning() const { return predictor ? predictor->is_online_learning() : false; }

    void learn() {
        if (predictor) {
            predictor->learn();
        }
    }

    bool is_layer_cache_initialized() const {
        return predictor ? predictor->is_layer_cache_initialized() : false;
    }

  private:
    std::unique_ptr<LSTMPredictor> predictor;

    // Pre-allocated vectors for update
    std::vector<std::vector<std::vector<float>>> update_input;
    std::vector<float> update_target;
    std::vector<float> update_pred;
    LSTMPredictor::LSTMOutput update_output;

    int lookback_len;
    int prediction_len;
};
#endif // NORMAL_DATA_PREDICTOR_HPP
