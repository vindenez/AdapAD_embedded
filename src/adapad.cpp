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

bool AdapAD::is_anomalous(float observed_val) {
    // Add the new observed value to observed_vals
    observed_vals.push_back(observed_val);

    // Ensure we have enough data for the lookback length
    if (observed_vals.size() < predictor_config.lookback_len) {
        std::cerr << "Not enough data to perform prediction. Need at least " << predictor_config.lookback_len << " values." << std::endl;
        return false;
    }

    // Get the input sequence for the LSTM (last lookback_len values)
    std::vector<float> input_data(observed_vals.end() - predictor_config.lookback_len, observed_vals.end());

    // Ensure input size matches the LSTM predictor's expected input size
    if (input_data.size() != data_predictor.get_input_size()) {
        std::cerr << "Error: Input size mismatch in is_anomalous(). Expected " 
                  << data_predictor.get_input_size() << " but got " << input_data.size() << std::endl;
        return false;  // Assuming false if we cannot process the input
    }

    // Predict a normal value
    std::vector<float> predicted_vals = data_predictor.predict(input_data);
    if (predicted_vals.empty()) {
        std::cerr << "Error: Predicted values are empty after forward pass." << std::endl;
        return false;
    }

    float predicted_val = predicted_vals[0];
    float error = observed_val - predicted_val;

    // Debug output
    std::cout << "Observed Value: " << observed_val << ", Predicted Value: " << predicted_val << ", Error: " << error << std::endl;

    // Update the generator based on the observed error
    if (predictive_errors.size() >= predictor_config.lookback_len) {
        std::cout << "Updating generator with new_input size: " << predictive_errors.size() << std::endl;
        generator.update(predictor_config.epoch_update, predictor_config.lr_update, predictive_errors, error);
    } else {
        std::cout << "Not updating generator: predictive_errors size (" << predictive_errors.size() 
                  << ") is less than lookback_len (" << predictor_config.lookback_len << ")" << std::endl;
    }

    predictive_errors.push_back(error);

    // Generate a threshold
    std::cout << "Generating threshold with prediction_errors size: " << predictive_errors.size() << std::endl;
    float threshold = generator.generate(predictive_errors, minimal_threshold);
    if (threshold == minimal_threshold) {
        std::cerr << "Warning: Returning minimal_threshold as generated threshold." << std::endl;
    }

    threshold = std::max(threshold, minimal_threshold);

    // Determine if the observation is anomalous
    bool is_anomalous = (std::abs(error) > threshold);
    std::cout << "Threshold: " << threshold << ", Anomalous: " << (is_anomalous ? "Yes" : "No") << std::endl;

    // Log the result
    if (f_log.is_open()) {
        f_log << observed_val << "," << predicted_val << ","
              << predicted_val - threshold << "," << predicted_val + threshold << ","
              << is_anomalous << "," << error << "," << threshold << "\n";
    }

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
