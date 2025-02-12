#include "model_state.hpp"
#include <stdexcept>

void ModelState::save_state(const std::string& filename,
                          const std::string& model_name,
                          const std::vector<LSTMPredictor::LSTMLayer>& weights,
                          const std::pair<std::vector<float>, std::vector<float>>& state) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Write model name
    size_t name_size = model_name.size();
    file.write(reinterpret_cast<const char*>(&name_size), sizeof(size_t));
    file.write(model_name.c_str(), name_size);

    // Write number of layers
    size_t num_layers = weights.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));

    // Write each layer
    for (const auto& layer : weights) {
        write_lstm_layer(file, layer);
    }

    // Write states
    write_vector(file, state.first);  // hidden state
    write_vector(file, state.second); // cell state
}

std::pair<std::vector<LSTMPredictor::LSTMLayer>, 
          std::pair<std::vector<float>, std::vector<float>>> 
ModelState::load_state(const std::string& filename, const std::string& model_name) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    // Read and verify model name
    size_t name_size;
    file.read(reinterpret_cast<char*>(&name_size), sizeof(size_t));
    std::string stored_name(name_size, '\0');
    file.read(&stored_name[0], name_size);
    
    if (stored_name != model_name) {
        throw std::runtime_error("Model name mismatch: expected " + model_name + 
                               ", found " + stored_name);
    }

    // Read number of layers
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));

    // Read layers
    std::vector<LSTMPredictor::LSTMLayer> weights;
    weights.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        weights.push_back(read_lstm_layer(file));
    }

    // Read states
    auto h_state = read_vector(file);
    auto c_state = read_vector(file);

    return {weights, {h_state, c_state}};
}

void ModelState::write_vector(std::ofstream& file, const std::vector<float>& vec) {
    size_t size = vec.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
}

std::vector<float> ModelState::read_vector(std::ifstream& file) {
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    std::vector<float> vec(size);
    file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    
    return vec;
}

void ModelState::write_lstm_layer(std::ofstream& file, const LSTMPredictor::LSTMLayer& layer) {
    // Write weight_ih
    size_t rows = layer.weight_ih.size();
    size_t cols = layer.weight_ih[0].size();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (const auto& row : layer.weight_ih) {
        file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
    }

    // Write weight_hh
    rows = layer.weight_hh.size();
    cols = layer.weight_hh[0].size();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (const auto& row : layer.weight_hh) {
        file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
    }

    // Write biases
    write_vector(file, layer.bias_ih);
    write_vector(file, layer.bias_hh);
}

LSTMPredictor::LSTMLayer ModelState::read_lstm_layer(std::ifstream& file) {
    LSTMPredictor::LSTMLayer layer;

    // Read weight_ih
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    layer.weight_ih.resize(rows, std::vector<float>(cols));
    for (auto& row : layer.weight_ih) {
        file.read(reinterpret_cast<char*>(row.data()), cols * sizeof(float));
    }

    // Read weight_hh
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    layer.weight_hh.resize(rows, std::vector<float>(cols));
    for (auto& row : layer.weight_hh) {
        file.read(reinterpret_cast<char*>(row.data()), cols * sizeof(float));
    }

    // Read biases
    layer.bias_ih = read_vector(file);
    layer.bias_hh = read_vector(file);

    return layer;
}