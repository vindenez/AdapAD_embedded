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

        AdapAD adap_ad(predictor_config, value_range_config, minimal_threshold);

        // Load training data first
        std::vector<DataPoint> training_data = load_csv_values(config::data_source_path);
        if (training_data.empty()) {
            throw std::runtime_error("Failed to load training data");
        }

        // Extract training values
        std::vector<float> training_values;
        for (const auto& point : training_data) {
            training_values.push_back(point.value);
        }

        // Train on the training data
        std::cout << "TRAINING ON TRAINING DATA..." << std::endl;
        adap_ad.set_training_data(training_values);
        std::cout << "TRAINING COMPLETE" << std::endl;

        // Now load and process validation data
        std::vector<DataPoint> validation_data = load_csv_values(config::data_val_path);
        if (validation_data.empty()) {
            throw std::runtime_error("Failed to load validation data");
        }

        std::cout << "STARTING VALIDATION..." << std::endl;
        
        std::vector<bool> predictions;
        std::vector<bool> actual_labels;

        // Process each validation point
        for (const auto& data_point : validation_data) {
            float measured_value = data_point.value;
            bool actual_anomaly = data_point.is_anomaly;
            
            bool predicted_anomaly = adap_ad.process(measured_value, actual_anomaly);
            predictions.push_back(predicted_anomaly);
            actual_labels.push_back(actual_anomaly);
            adap_ad.clean();
        }

        // Calculate metrics on validation data
        if (!predictions.empty()) {
            Metrics metrics = calculate_metrics(predictions, actual_labels);
            std::cout << "Validation Results:" << std::endl;
            std::cout << "Accuracy: " << metrics.accuracy << std::endl;
            std::cout << "Precision: " << metrics.precision << std::endl;
            std::cout << "Recall: " << metrics.recall << std::endl;
            std::cout << "F1 Score: " << metrics.f1_score << std::endl;
        }

        std::cout << "Done! Check result at " << adap_ad.get_log_filename() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
