#include "yaml_handler.hpp"
#include <iostream>
#include <iomanip>

void print_config(const std::map<std::string, std::string>& config) {
    for (const auto& pair : config) {
        std::cout << std::setw(50) << std::left << pair.first 
                  << " -> " << pair.second << std::endl;
    }
}

int main() {
    auto config = SimpleYAML::parse("config.yaml");
    
    // Test some key paths that should exist
    std::vector<std::string> test_keys = {
        "data.source",
        "data.paths.training",
        "data.sources.Tide_pressure.epochs.train",
        "data.sources.Tide_pressure.bounds.lower",
        "model.lstm.size",
        "training.epochs.train",
        "anomaly_detection.threshold_multiplier"
    };
    
    bool all_passed = true;
    
    std::cout << "Testing specific keys:\n";
    for (const auto& key : test_keys) {
        if (config.find(key) != config.end()) {
            std::cout << "✓ Found " << key << " = " << config[key] << std::endl;
        } else {
            std::cout << "✗ Missing key: " << key << std::endl;
            all_passed = false;
        }
    }
    
    std::cout << "\nComplete config dump:\n";
    print_config(config);
    
    return all_passed ? 0 : 1;
} 