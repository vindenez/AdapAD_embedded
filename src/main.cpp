#include "normal_data_predictor.hpp"
#include "anomalous_threshold_generator.hpp"
#include "adapad.hpp"
#include "json_loader.hpp"
#include "config.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

std::vector<float> load_csv_values(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    std::vector<float> values;
    std::string line;
    
    // Skip the header line
    std::getline(file, line);

    // Read each line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp, value_str, is_anomaly;

        // Parse line - assuming the structure timestamp,value,is_anomaly
        std::getline(ss, timestamp, ',');
        std::getline(ss, value_str, ',');
        std::getline(ss, is_anomaly, ',');

        try {
            float value = std::stof(value_str);
            values.push_back(value);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing value: " << e.what() << std::endl;
        }
    }

    return values;
}

int main() {
    try {
        // Load weights and biases from JSON file
        auto weights = load_all_weights("weights/lstm_weights.json");
        auto biases = load_all_biases("weights/lstm_weights.json");

        // Initialize configuration
        PredictorConfig predictor_config = init_predictor_config();
        float minimal_threshold;
        ValueRangeConfig value_range_config = init_value_range_config(data_source, minimal_threshold);

        if (minimal_threshold == 0) {
            throw std::runtime_error("Minimal threshold must be set.");
        }

        // Create AdapAD instance
        AdapAD adap_ad(predictor_config, value_range_config, minimal_threshold);

        // Load training data
        std::vector<float> training_data = load_csv_values("data/Tide_pressure.csv");
        adap_ad.set_training_data(training_data);

        // Train the model
        adap_ad.train();

        // Validation Stage
        std::vector<float> validation_data = load_csv_values("data/Tide_pressure.validation_stage.csv");
        for (float val : validation_data) {
            bool is_anomalous = adap_ad.is_anomalous(val);
            if (is_anomalous) {
                std::cout << "Validation: Anomalous value detected: " << val << std::endl;
            } else {
                std::cout << "Validation: Normal value: " << val << std::endl;
            }
        }

        // Benchmark Stage
        std::vector<float> benchmark_data = load_csv_values("data/Tide_pressure.benchmark_stage.csv");
        for (float val : benchmark_data) {
            bool is_anomalous = adap_ad.is_anomalous(val);
            if (is_anomalous) {
                std::cout << "Benchmark: Anomalous value detected: " << val << std::endl;
            } else {
                std::cout << "Benchmark: Normal value: " << val << std::endl;
            }
        }

        adap_ad.clean();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
