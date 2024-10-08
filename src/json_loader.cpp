#include "json_loader.hpp"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>

using json = nlohmann::json;

// Load all LSTM weights from JSON file
std::unordered_map<std::string, std::vector<std::vector<float>>> load_all_weights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    json json_data;
    try {
        file >> json_data;
    } catch (const json::parse_error& e) {
        std::cerr << "Error: JSON parsing failed. " << e.what() << std::endl;
        return {};
    }

    std::unordered_map<std::string, std::vector<std::vector<float>>> all_weights;

    for (const auto& [key, value] : json_data.items()) {
        if (key.find("lstm.weight") != std::string::npos || key.find("fc.weight") != std::string::npos) {
            if (value.is_array()) {
                all_weights[key] = value.get<std::vector<std::vector<float>>>();
            }
        }
    }

    return all_weights;
}

// Load all LSTM biases from JSON file
std::unordered_map<std::string, std::vector<float>> load_all_biases(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    json json_data;
    try {
        file >> json_data;
    } catch (const json::parse_error& e) {
        std::cerr << "Error: JSON parsing failed. " << e.what() << std::endl;
        return {};
    }

    std::unordered_map<std::string, std::vector<float>> all_biases;

    for (const auto& [key, value] : json_data.items()) {
        if (key.find("lstm.bias") != std::string::npos || key.find("fc.bias") != std::string::npos) {
            if (value.is_array()) {
                all_biases[key] = value.get<std::vector<float>>();
            }
        }
    }

    return all_biases;
}

std::vector<std::vector<float>> load_weights(const std::string& filename, const std::string& key) {
    std::unordered_map<std::string, std::vector<std::vector<float>>> all_weights = load_all_weights(filename);
    if (all_weights.find(key) != all_weights.end()) {
        return all_weights[key];
    } else {
        std::cerr << "Error: Could not find key " << key << " in weights file." << std::endl;
        return {};
    }
}

std::vector<float> load_bias(const std::string& filename, const std::string& key) {
    std::unordered_map<std::string, std::vector<float>> all_biases = load_all_biases(filename);
    if (all_biases.find(key) != all_biases.end()) {
        return all_biases[key];
    } else {
        std::cerr << "Error: Could not find key " << key << " in biases file." << std::endl;
        return {};
    }
}