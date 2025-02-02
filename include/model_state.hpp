#ifndef MODEL_STATE_HPP
#define MODEL_STATE_HPP

#include "lstm_predictor.hpp"
#include <string>
#include <vector>
#include <fstream>

class ModelState {
public:
    static void save_state(const std::string& filename, 
                          const std::string& model_name,
                          const std::vector<LSTMPredictor::LSTMLayer>& weights,
                          const std::pair<std::vector<float>, std::vector<float>>& state);
    
    static std::pair<std::vector<LSTMPredictor::LSTMLayer>, 
                    std::pair<std::vector<float>, std::vector<float>>> 
    load_state(const std::string& filename, const std::string& model_name);

private:
    static void write_vector(std::ofstream& file, const std::vector<float>& vec);
    static std::vector<float> read_vector(std::ifstream& file);
    static void write_lstm_layer(std::ofstream& file, const LSTMPredictor::LSTMLayer& layer);
    static LSTMPredictor::LSTMLayer read_lstm_layer(std::ifstream& file);
};

#endif // MODEL_STATE_HPP 