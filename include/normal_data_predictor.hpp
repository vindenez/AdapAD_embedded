#ifndef NORMAL_DATA_PREDICTOR_HPP
#define NORMAL_DATA_PREDICTOR_HPP

#include "lstm_predictor.hpp"
#include <vector>
#include <memory>
#include "lstm_predictor.hpp"
#include <fstream>

class NormalDataPredictor {
public:
    NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len);
    
    std::pair<std::vector<std::vector<float>>, std::vector<float>>
    create_sliding_windows(const std::vector<float>& data);
    
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
    train(int epoch, float lr, const std::vector<float>& data2learn);
    
    float predict(const std::vector<std::vector<std::vector<float>>>& observed);
    
    void update(int epoch_update, float lr_update,
                const std::vector<std::vector<std::vector<float>>>& past_observations,
                const std::vector<float>& recent_observation);

    void reset_states();
    void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                   const std::vector<float>& target,
                   const LSTMPredictor::LSTMOutput& lstm_output,
                   float learning_rate) {
        predictor->train_step(x, target, lstm_output, learning_rate);
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

    void clear_temporary_cache() {
        if (predictor) {
            predictor->clear_temporary_cache();
        }
    }

    void clear_training_state();

    // Add this method to expose training mode status
    bool is_training() const { 
        return predictor ? predictor->is_training() : false; 
    }

    bool is_online_learning() const {
        return predictor ? predictor->is_online_learning() : false;
    }

    void learn() {
        if (predictor) {
            predictor->learn();
        }
    }

    // Helper function to get current memory usage
    size_t get_current_memory() const;

    bool is_layer_cache_initialized() const { 
        return predictor ? predictor->is_layer_cache_initialized() : false; 
    }

private:
    int lookback_len;
    int prediction_len;
    std::unique_ptr<LSTMPredictor> predictor;
    
    // Pre-allocated vectors for update
    std::vector<std::vector<std::vector<float>>> update_input;
    std::vector<float> update_target;
    std::vector<float> update_pred;
    LSTMPredictor::LSTMOutput update_output;
};
#endif // NORMAL_DATA_PREDICTOR_HPP
