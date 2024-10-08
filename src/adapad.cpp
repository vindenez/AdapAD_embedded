#include "adapad.hpp"
#include "json_loader.hpp"
#include "config.hpp"  // Include config to access global variables like update_G_epoch and update_G_lr
#include <iostream>
#include <cmath>

AdapAD::AdapAD(const PredictorConfig& predictor_config, const ValueRangeConfig& value_range_config, float minimal_threshold)
    : data_predictor(load_all_weights("weights/lstm_weights.json"), load_all_biases("weights/lstm_weights.json")),
      generator(predictor_config.lookback_len, predictor_config.prediction_len, value_range_config.lower_bound, value_range_config.upper_bound),
      minimal_threshold(minimal_threshold),
      predictor_config(predictor_config) {

    // Initialize value ranges
    observed_vals.reserve(100);  // Reserve some space for observed values
    thresholds.push_back(minimal_threshold);

    // Log file setup
    f_log.open("anomalous_detection_log.csv");
    if (!f_log.is_open()) {
        std::cerr << "Error: Could not open log file." << std::endl;
    }
    f_log << "Observed,Predicted,LowerBound,UpperBound,Anomalous,Error,Threshold\n";
}

void AdapAD::set_training_data(const std::vector<float>& data) {
    observed_vals = data;
}

void AdapAD::train() {
    // Train the anomaly threshold generator based on prediction errors
    for (size_t i = 0; i < observed_vals.size(); ++i) {
        float predicted_val = data_predictor.predict({observed_vals[i]})[0];
        predicted_vals.push_back(predicted_val);

        float error = observed_vals[i] - predicted_val;
        predictive_errors.push_back(error);

        if (predictive_errors.size() >= predictor_config.lookback_len) {
            // Update the generator with predictive errors
            generator.update(predictor_config.epoch_update, predictor_config.lr_update, predictive_errors, error);
        }
    }

    std::cout << "Trained AdapAD components." << std::endl;
}

bool AdapAD::is_anomalous(float observed_val) {
    // Predict a normal value
    float predicted_val = data_predictor.predict({observed_val})[0];
    float error = observed_val - predicted_val;
    float threshold = generator.generate(predictive_errors, minimal_threshold);
    threshold = std::max(threshold, minimal_threshold);

    // Determine if the observation is anomalous
    bool is_anomalous = (std::abs(error) > threshold);

    // Log the result
    f_log << observed_val << "," << predicted_val << ","
          << predicted_val - threshold << "," << predicted_val + threshold << ","
          << is_anomalous << "," << error << "," << threshold << "\n";

    return is_anomalous;
}

void AdapAD::clean() {
    // This function can clean up values if needed to keep memory use minimal
    if (observed_vals.size() > 100) {
        observed_vals.erase(observed_vals.begin(), observed_vals.begin() + 10);
    }
    if (predicted_vals.size() > 100) {
        predicted_vals.erase(predicted_vals.begin(), predicted_vals.begin() + 10);
    }
    if (thresholds.size() > 100) {
        thresholds.erase(thresholds.begin(), thresholds.begin() + 10);
    }
}

void AdapAD::log_results() {
    if (f_log.is_open()) {
        f_log.flush();
    }
}
