#ifndef NORMAL_DATA_PREDICTOR_HPP
#define NORMAL_DATA_PREDICTOR_HPP

#include "lstm_predictor.hpp"
#include <unordered_map>
#include <vector>
#include <string>

class NormalDataPredictor {
public:
    NormalDataPredictor(const std::unordered_map<std::string, std::vector<std::vector<float>>>& weights,
                        const std::unordered_map<std::string, std::vector<float>>& biases);

    std::vector<float> predict(const std::vector<float>& input);
    int get_input_size() const;
    void update(int num_epochs, float learning_rate, const std::vector<float>& past_observations, float recent_observation);

    int get_hidden_size() const {
        return lstm_layers.empty() ? 0 : lstm_layers[0].get_hidden_size();
    }

private:
    std::vector<LSTMPredictor> lstm_layers;
    std::vector<std::vector<float>> fc_weight;
    std::vector<float> fc_bias;
    int input_size;

    std::vector<float> fully_connected(const std::vector<float>& input);
};

#endif // NORMAL_DATA_PREDICTOR_HPP
