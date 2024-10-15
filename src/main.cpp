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
        std::stringstream ss(line);
        std::string timestamp, value_str, is_anomaly_str;

        std::getline(ss, timestamp, ',');
        std::getline(ss, value_str, ',');
        std::getline(ss, is_anomaly_str, ',');

        try {
            float value = std::stof(value_str);
            bool is_anomaly = (is_anomaly_str == "1");
            data_points.push_back({timestamp, value, is_anomaly});
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
        // Load configuration
        PredictorConfig predictor_config = init_predictor_config();
        float minimal_threshold;
        ValueRangeConfig value_range_config = init_value_range_config(config::data_source, minimal_threshold);

        if (minimal_threshold == 0) {
            throw std::runtime_error("It is mandatory to set a minimal threshold");
        }

        // Load data
        std::vector<DataPoint> data_production = load_csv_values(config::data_source_path);
        size_t len_data_subject = data_production.size();

        // Create NormalDataPredictor first
        NormalDataPredictor data_predictor(weights, biases);

        // Then create AdapAD instance
        AdapAD adap_ad(predictor_config, value_range_config, minimal_threshold, data_predictor, {});
        std::cout << "GATHERING DATA FOR TRAINING... " << predictor_config.train_size << std::endl;

        std::vector<float> observed_data;

        for (size_t data_idx = 0; data_idx < len_data_subject; ++data_idx) {
            const auto& data_point = data_production[data_idx];
            float measured_value = data_point.value;
            bool actual_anomaly = data_point.is_anomaly;

            observed_data.push_back(measured_value);
            size_t observed_data_sz = observed_data.size();

            // Perform warmup training or make a decision
            if (observed_data_sz == predictor_config.train_size) {
                adap_ad.set_training_data(observed_data);
                std::cout << "------------STARTING TO MAKE DECISION------------" << std::endl;
            } else if (observed_data_sz > predictor_config.train_size) {
                bool is_anomalous_ret = adap_ad.is_anomalous(measured_value, actual_anomaly);
                adap_ad.clean();
            } else {
                std::cout << observed_data.size() << "/" << predictor_config.train_size << " to warmup training" << std::endl;
            }
        }

        std::cout << "Done! Check result at " << adap_ad.get_log_filename() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
