#include "normal_data_predictor.hpp"
#include "anomalous_threshold_generator.hpp"
#include "adapad.hpp"
#include "json_loader.hpp"
#include "config.hpp"
#include "lstm_predictor.hpp"  // Make sure to include this header
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

std::vector<float> load_csv_values(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    std::vector<float> values;
    std::string line;

    // Skip the header line
    if (!std::getline(file, line)) {
        std::cerr << "Error: Failed to read header from file " << filename << std::endl;
        return {};
    }

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

    if (values.empty()) {
        std::cerr << "Error: No valid data loaded from file " << filename << std::endl;
    }

    return values;
}

struct DataPoint {
    float value;
    bool is_anomaly;
};

std::vector<DataPoint> load_csv_values_with_labels(const std::string& filename) {
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
            data_points.push_back({value, is_anomaly});
        } catch (const std::exception& e) {
            std::cerr << "Error parsing value: " << e.what() << std::endl;
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
        // Load weights and biases from JSON file
        auto weights = load_all_weights("weights/lstm_weights.json");
        auto biases = load_all_biases("weights/lstm_weights.json");

        // Initialize configuration
        PredictorConfig predictor_config = init_predictor_config();
        float minimal_threshold;
        ValueRangeConfig value_range_config = init_value_range_config(config::data_source, minimal_threshold);

        if (minimal_threshold == 0) {
            throw std::runtime_error("Minimal threshold must be set.");
        }

        // Create LSTMPredictor with correct dimensions
        LSTMPredictor lstm_predictor(
            predictor_config.input_size,
            predictor_config.hidden_size,
            predictor_config.num_layers,
            predictor_config.lookback_len
        );

        
        // Create AdapAD instance with the LSTMPredictor
        AdapAD adap_ad(predictor_config, value_range_config, minimal_threshold, lstm_predictor);

        // Load training data
        std::vector<float> training_data = load_csv_values("data/Tide_pressure.csv");
        if (training_data.empty()) {
            throw std::runtime_error("Training data could not be loaded. Check the file path and contents.");
        }

        adap_ad.set_training_data(training_data);

        // Load validation data with labels
        std::vector<DataPoint> validation_data = load_csv_values_with_labels("data/Tide_pressure.validation_stage.csv");
        if (validation_data.empty()) {
            throw std::runtime_error("Validation data could not be loaded. Check the file path and contents.");
        }

        std::vector<bool> predictions;
        std::vector<bool> actual_labels;

        for (const auto& data_point : validation_data) {
            bool is_anomalous = adap_ad.is_anomalous(data_point.value);
            predictions.push_back(is_anomalous);
            actual_labels.push_back(data_point.is_anomaly);

            if (is_anomalous) {
                std::cout << "Validation: Anomalous value detected: " << data_point.value << std::endl;
            } else {
                std::cout << "Validation: Normal value: " << data_point.value << std::endl;
            }
        }

        Metrics validation_metrics = calculate_metrics(predictions, actual_labels);

        std::cout << "Validation Metrics:" << std::endl;
        std::cout << "Accuracy: " << validation_metrics.accuracy << std::endl;
        std::cout << "Precision: " << validation_metrics.precision << std::endl;
        std::cout << "Recall: " << validation_metrics.recall << std::endl;
        std::cout << "F1-score: " << validation_metrics.f1_score << std::endl;

        // Benchmark Stage
        std::vector<float> benchmark_data = load_csv_values("data/Tide_pressure.benchmark_stage.csv");
        if (benchmark_data.empty()) {
            throw std::runtime_error("Benchmark data could not be loaded. Check the file path and contents.");
        }
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
        return 1;
    }

    return 0;
}