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

        // Initialize AdapAD
        AdapAD adap_ad(predictor_config, value_range_config, minimal_threshold);

        // Load validation data only (we'll use part of it for training)
        std::vector<DataPoint> data = load_csv_values("data/tide_pressure.validation_stage.csv");
        if (data.empty()) {
            throw std::runtime_error("Failed to load data");
        }

        std::cout << "GATHERING DATA FOR TRAINING..." << predictor_config.train_size << std::endl;
        
        std::vector<bool> predictions;
        std::vector<bool> actual_labels;
        std::vector<float> observed_data;

        // Process each point sequentially
        for (const auto& data_point : data) {
            float measured_value = data_point.value;
            bool actual_anomaly = data_point.is_anomaly;
            observed_data.push_back(measured_value);
            size_t observed_data_sz = observed_data.size();

            // Perform warmup training or make a decision
            if (observed_data_sz == predictor_config.train_size) {
                std::cout << "Initial training with " << observed_data_sz << " points" << std::endl;
                adap_ad.set_training_data(observed_data);
                std::cout << "------------STARTING TO MAKE DECISION------------" << std::endl;
            }
            else if (observed_data_sz > predictor_config.train_size) {
                bool predicted_anomaly = adap_ad.is_anomalous(measured_value, actual_anomaly);
                predictions.push_back(predicted_anomaly);
                actual_labels.push_back(actual_anomaly);
                adap_ad.clean();
            }
            else {
                std::cout << observed_data_sz << "/" << predictor_config.train_size 
                         << " to warmup training" << std::endl;
            }
        }

        // Calculate metrics only on the validation portion
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
