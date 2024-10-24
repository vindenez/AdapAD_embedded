#include "normal_data_predictor.hpp"
#include "anomalous_threshold_generator.hpp"
#include "adapad.hpp"
#include "json_loader.hpp"
#include "config.hpp"
#include "lstm_predictor.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>

struct DataPoint {
    std::string timestamp;
    float value;
    bool is_anomaly;  
};

std::vector<DataPoint> load_csv_values(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    std::vector<DataPoint> data_points;
    std::string line;

    // Skip the header line
    std::getline(file, line);

    // Read each line
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string timestamp, value_str, is_anomaly_str;

        // Parse CSV format
        std::getline(ss, timestamp, ',');
        std::getline(ss, value_str, ',');
        std::getline(ss, is_anomaly_str, ',');

        try {
            float value = std::stof(value_str);
            
            // Trim whitespace from is_anomaly_str
            is_anomaly_str.erase(0, is_anomaly_str.find_first_not_of(" \n\r\t"));
            is_anomaly_str.erase(is_anomaly_str.find_last_not_of(" \n\r\t") + 1);
            
            // Convert string to bool, checking for both "1" and "1\n" etc.
            bool is_anomaly = (is_anomaly_str == "1" || is_anomaly_str == "1\n" || is_anomaly_str == "1\r" || is_anomaly_str == "1\r\n");

            // Debug print with more detail
            std::cout << "Loaded: " << timestamp << ", value=" << value 
                     << ", is_anomaly=" << is_anomaly 
                     << " (raw='" << is_anomaly_str << "', length=" << is_anomaly_str.length() 
                     << ", hex=";
            
            // Print hex values of the string for debugging
            for (char c : is_anomaly_str) {
                std::cout << std::hex << (int)c << " ";
            }
            std::cout << ")" << std::endl;

            data_points.push_back({timestamp, value, is_anomaly});
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << "\nError: " << e.what() << std::endl;
            continue;
        }
    }

    return data_points;
}

struct Metrics {
    float accuracy;
    float precision;
    float recall;
    float f1_score;
};

Metrics calculate_metrics(const std::vector<bool>& predictions, const std::vector<bool>& actual_labels) {
    int true_positives = 0, false_positives = 0, true_negatives = 0, false_negatives = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] && actual_labels[i]) true_positives++;
        else if (predictions[i] && !actual_labels[i]) false_positives++;
        else if (!predictions[i] && !actual_labels[i]) true_negatives++;
        else if (!predictions[i] && actual_labels[i]) false_negatives++;
    }

    float accuracy = static_cast<float>(true_positives + true_negatives) / predictions.size();
    float precision = true_positives / static_cast<float>(true_positives + false_positives);
    float recall = true_positives / static_cast<float>(true_positives + false_negatives);
    float f1_score = 2 * (precision * recall) / (precision + recall);

    return {accuracy, precision, recall, f1_score};
}

int main() {
    try {
        PredictorConfig predictor_config = init_predictor_config();
        float minimal_threshold;
        ValueRangeConfig value_range_config = init_value_range_config(config::data_source, minimal_threshold);

        if (minimal_threshold == 0) {
            throw std::runtime_error("It is mandatory to set a minimal threshold");
        }

        // Initialize AdapAD
        AdapAD adap_ad(predictor_config, value_range_config, config::minimal_threshold);

        // Load training data
        std::vector<DataPoint> training_data = load_csv_values("data/tide_pressure.csv");
        if (training_data.empty()) {
            throw std::runtime_error("Failed to load training data");
        }

        std::cout << "TRAINING ON TIDE_PRESSURE.CSV..." << std::endl;

        // Prepare training data
        std::vector<float> training_values;
        for (const auto& data_point : training_data) {
            training_values.push_back(data_point.value);
        }

        // Set training data and train
        adap_ad.set_training_data(training_values);

        std::cout << "Training complete. Starting validation..." << std::endl;

        // Load validation data
        std::vector<DataPoint> validation_data = load_csv_values("data/tide_pressure.validation_stage.csv");
        if (validation_data.empty()) {
            throw std::runtime_error("Failed to load validation data");
        }

        // Validation phase
        std::vector<bool> predictions;
        std::vector<bool> actual_labels;

        for (const auto& data_point : validation_data) {
            float measured_value = data_point.value;
            bool actual_anomaly = data_point.is_anomaly;

            // Debug output for all data points
            std::cout << "Processing: " << data_point.timestamp << ", " << measured_value 
                      << ", is_anomaly: " << (actual_anomaly ? "true" : "false") << std::endl;

            // Process each data point
            bool predicted_anomaly = adap_ad.is_anomalous(measured_value, actual_anomaly);

            predictions.push_back(predicted_anomaly);
            actual_labels.push_back(actual_anomaly);

            adap_ad.clean();
        }

        // Calculate metrics
        Metrics metrics = calculate_metrics(predictions, actual_labels);

        // Print metrics
        std::cout << "Validation Results:" << std::endl;
        std::cout << "Accuracy: " << metrics.accuracy << std::endl;
        std::cout << "Precision: " << metrics.precision << std::endl;
        std::cout << "Recall: " << metrics.recall << std::endl;
        std::cout << "F1 Score: " << metrics.f1_score << std::endl;

        std::cout << "Done! Check result at " << adap_ad.get_log_filename() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
