#ifndef NORMAL_DATA_PREDICTOR_HPP
#define NORMAL_DATA_PREDICTOR_HPP

#include "lstm_predictor.hpp"
#include <unordered_map>
#include <vector>
#include <string>

class NormalDataPredictor {
public:
    // Constructor to initialize the NormalDataPredictor with LSTM layer weights and biases for each gate
    NormalDataPredictor(const std::unordered_map<std::string, std::vector<std::vector<float>>>& weights,
                        const std::unordered_map<std::string, std::vector<float>>& biases);

    // Function to perform prediction with the given input
    std::vector<float> predict(const std::vector<float>& input);

    // Function to get the input size for the first LSTM layer
    int get_input_size() const;

private:
    // Vector to hold all LSTM layers
    std::vector<LSTMPredictor> lstm_layers;

    // Fully connected layer weights and biases
    std::vector<std::vector<float>> fc_weight;
    std::vector<float> fc_bias;

    int input_size;

    // Function to perform a fully connected layer operation
    std::vector<float> fully_connected(const std::vector<float>& input);
};

#endif // NORMAL_DATA_PREDICTOR_HPP
