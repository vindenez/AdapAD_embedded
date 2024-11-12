#include "json_loader.hpp"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>

using json = nlohmann::json;

std::unordered_map<std::string, std::vector<std::vector<float>>> load_all_weights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return {};
    } else {
        std::cout << "Successfully opened file '" << filename << "'" << std::endl;
    }

    json json_data;
    try {
        file >> json_data;
        std::cout << "Successfully parsed JSON data from '" << filename << "'" << std::endl;
    } catch (const json::parse_error& e) {
        std::cerr << "Error: JSON parsing failed. " << e.what() << std::endl;
        return {};
    }

    std::unordered_map<std::string, std::vector<std::vector<float>>> all_weights;

    std::cout << "Keys in JSON data:" << std::endl;
    for (const auto& [key, value] : json_data.items()) {
        std::cout << key << std::endl;
    }

    for (const auto& [key, value] : json_data.items()) {
        if (key.find("lstm.weight") != std::string::npos || key.find("fc.weight") != std::string::npos) {
            if (value.is_array()) {
                try {
                    std::vector<std::vector<float>> weight_matrix = value.get<std::vector<std::vector<float>>>();
                    all_weights[key] = weight_matrix;

                    size_t rows = weight_matrix.size();
                } catch (const json::exception& e) {
                    std::cerr << "Error parsing weight matrix for key '" << key << "': " << e.what() << std::endl;
                }
            } else {
                std::cerr << "Value for key '" << key << "' is not an array." << std::endl;
            }
        }
    }

    return all_weights;
}

std::unordered_map<std::string, std::vector<float>> load_all_biases(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return {};
    } else {
        std::cout << "Successfully opened file '" << filename << "'" << std::endl;
    }

    json json_data;
    try {
        file >> json_data;
        std::cout << "Successfully parsed JSON data from '" << filename << "'" << std::endl;
    } catch (const json::parse_error& e) {
        std::cerr << "Error: JSON parsing failed. " << e.what() << std::endl;
        return {};
    }

    std::unordered_map<std::string, std::vector<float>> all_biases;

    std::cout << "Keys in JSON data:" << std::endl;
    for (const auto& [key, value] : json_data.items()) {
        std::cout << key << std::endl;
    }

    for (const auto& [key, value] : json_data.items()) {
        if (key.find("lstm.bias") != std::string::npos || key.find("fc.bias") != std::string::npos) {
            if (value.is_array()) {
                try {
                    std::vector<float> bias_vector = value.get<std::vector<float>>();
                    all_biases[key] = bias_vector;

                    size_t size = bias_vector.size();
                    std::cout << "Loaded bias '" << key << "' with size: " << size << std::endl;
                } catch (const json::exception& e) {
                    std::cerr << "Error parsing bias vector for key '" << key << "': " << e.what() << std::endl;
                }
            } else {
                std::cerr << "Value for key '" << key << "' is not an array." << std::endl;
            }
        }
    }

    return all_biases;
}

// Load specific weights by key
std::vector<std::vector<float>> load_weights(const std::string& filename, const std::string& key) {
    std::unordered_map<std::string, std::vector<std::vector<float>>> all_weights = load_all_weights(filename);
    if (all_weights.find(key) != all_weights.end()) {
        std::cout << "Successfully loaded weight for key: '" << key << "'" << std::endl;
        return all_weights[key];
    } else {
        std::cerr << "Error: Could not find key '" << key << "' in weights file." << std::endl;
        std::cerr << "Available keys are:" << std::endl;
        for (const auto& [k, v] : all_weights) {
            std::cout << k << std::endl;
        }
        return {};
    }
}

// Load specific bias by key
std::vector<float> load_bias(const std::string& filename, const std::string& key) {
    std::unordered_map<std::string, std::vector<float>> all_biases = load_all_biases(filename);
    if (all_biases.find(key) != all_biases.end()) {
        std::cout << "Successfully loaded bias for key: '" << key << "'" << std::endl;
        return all_biases[key];
    } else {
        std::cerr << "Error: Could not find key '" << key << "' in biases file." << std::endl;
        std::cerr << "Available keys are:" << std::endl;
        for (const auto& [k, v] : all_biases) {
            std::cout << k << std::endl;
        }
        return {};
    }
}
