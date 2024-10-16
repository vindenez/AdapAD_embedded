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
        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string timestamp, value_str, is_anomaly_str;

        std::getline(ss, timestamp, ',');
        std::getline(ss, value_str, ',');
        std::getline(ss, is_anomaly_str, ',');

        try {
            float value = std::stof(value_str);

            // Trim whitespace from is_anomaly_str
            is_anomaly_str.erase(0, is_anomaly_str.find_first_not_of(" \t\n\r\f\v"));
            is_anomaly_str.erase(is_anomaly_str.find_last_not_of(" \t\n\r\f\v") + 1);

            // Debug output to check the raw string
            std::cout << "Raw is_anomaly_str: '" << is_anomaly_str << "'" << std::endl;

            bool is_anomaly = (is_anomaly_str == "1");

            data_points.push_back({timestamp, value, is_anomaly});
            
            // Debug output for all rows
            std::cout << "CSV row: " << timestamp << ", " << value << ", is_anomaly: " << (is_anomaly ? "true" : "false") << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << "\nError: " << e.what() << std::endl;
        }
    }

    if (data_points.empty()) {
        std::cerr << "Error: No valid data loaded from file " << filename << std::endl;
    } else {
        std::cout << "Loaded " << data_points.size() << " data points from " << filename << std::endl;
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
        auto weights = load_all_weights("weights/lstm_weights.json");
        auto biases = load_all_biases("weights/lstm_weights.json");
        PredictorConfig predictor_config = init_predictor_config();
        float minimal_threshold;
        ValueRangeConfig value_range_config = init_value_range_config(config::data_source, minimal_threshold);

        if (minimal_threshold == 0) {
            throw std::runtime_error("It is mandatory to set a minimal threshold");
        }

        // Load training data
        std::vector<DataPoint> training_data = load_csv_values("data/tide_pressure.csv");
        if (training_data.empty()) {
            throw std::runtime_error("Failed to load training data");
        }

        NormalDataPredictor data_predictor(weights, biases);
        AnomalousThresholdGenerator threshold_generator(predictor_config.lookback_len, predictor_config.prediction_len, value_range_config.lower_bound, value_range_config.upper_bound);
        
        AdapAD adap_ad(predictor_config, value_range_config, minimal_threshold, data_predictor, {});
        std::cout << "TRAINING ON TIDE_PRESSURE.CSV..." << std::endl;

        // Training phase
        for (const auto& data_point : training_data) {
            float measured_value = data_point.value;
            bool actual_anomaly = data_point.is_anomaly;

            // Process each data point without logging
            adap_ad.is_anomalous(measured_value, actual_anomaly, false);
            adap_ad.clean();
        }

        std::cout << "Training complete. Starting validation..." << std::endl;

        // Open log file for validation phase
        adap_ad.open_log_file();

        // Load validation data
        std::vector<DataPoint> validation_data = load_csv_values("data/tide_pressure.validation_stage.csv");
        if (validation_data.empty()) {
            throw std::runtime_error("Failed to load validation data");
        }

        // Validation phase
        int true_positives = 0, false_positives = 0, true_negatives = 0, false_negatives = 0;

        for (const auto& data_point : validation_data) {
            float measured_value = data_point.value;
            bool actual_anomaly = data_point.is_anomaly;

            // Debug output for all data points
            std::cout << "Processing: " << data_point.timestamp << ", " << measured_value 
                      << ", is_anomaly: " << (actual_anomaly ? "true" : "false") << std::endl;

            // Process each data point and log results
            bool predicted_anomaly = adap_ad.is_anomalous(measured_value, actual_anomaly, true);

            if (predicted_anomaly && actual_anomaly) true_positives++;
            else if (predicted_anomaly && !actual_anomaly) false_positives++;
            else if (!predicted_anomaly && !actual_anomaly) true_negatives++;
            else if (!predicted_anomaly && actual_anomaly) false_negatives++;

            adap_ad.clean();
        }

        // Calculate and print metrics
        float accuracy = static_cast<float>(true_positives + true_negatives) / validation_data.size();
        float precision = static_cast<float>(true_positives) / (true_positives + false_positives);
        float recall = static_cast<float>(true_positives) / (true_positives + false_negatives);
        float f1_score = 2 * (precision * recall) / (precision + recall);

        std::cout << "Validation Results:" << std::endl;
        std::cout << "Accuracy: " << accuracy << std::endl;
        std::cout << "Precision: " << precision << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "F1 Score: " << f1_score << std::endl;

        std::cout << "Done! Check result at " << adap_ad.get_log_filename() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
