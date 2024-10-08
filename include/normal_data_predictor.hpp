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

private:
    std::vector<LSTMPredictor> lstm_layers;
    std::vector<std::vector<float>> fc_weight;
    std::vector<float> fc_bias;

    std::vector<float> fully_connected(const std::vector<float>& input);
};

#endif // NORMAL_DATA_PREDICTOR_HPP
