#include "adapad.hpp"
#include "json_loader.hpp"
#include "config.hpp" 
#include <iostream>
#include <cmath>
#include "config.hpp"

AdapAD::AdapAD(const PredictorConfig& predictor_config,
               const ValueRangeConfig& value_range_config,
               float minimal_threshold,
               NormalDataPredictor& data_predictor,
               const std::vector<float>& training_data)
    : data_predictor(data_predictor),
      generator(predictor_config.lookback_len,
                predictor_config.prediction_len,
                0.0f,  // lower_bound
                1.0f), // upper_bound
      minimal_threshold(minimal_threshold),
      predictor_config(predictor_config),
      value_range_config(value_range_config)
{
    try {
        observed_vals.reserve(predictor_config.lookback_len);
        predicted_vals.reserve(predictor_config.lookback_len);
        predictive_errors.reserve(predictor_config.lookback_len);
        thresholds.reserve(predictor_config.lookback_len);
        thresholds.push_back(minimal_threshold);

        // Log file setup
        f_log.open(config::log_file_path);
        if (!f_log.is_open()) {
            throw std::runtime_error("Could not open log file.");
        }
        f_log << "Observed,Predicted,LowerBound,UpperBound,PredictedAnomaly,ActualAnomaly,Error,Threshold\n";

        // Train the generator
        std::vector<float> normalized_training_data;
        for (const auto& val : training_data) {
            normalized_training_data.push_back(normalize_data(val));
        }
        generator.train(config::epoch_train, config::lr_train, normalized_training_data);
    } catch (const std::exception& e) {
        std::cerr << "Error in AdapAD constructor: " << e.what() << std::endl;
        throw;
    }
}

void AdapAD::set_training_data(const std::vector<float>& data) {
    // Clear existing observed values
    observed_vals.clear();

    // Normalize and store the new training data
    for (const auto& val : data) {
        observed_vals.push_back(normalize_data(val));
    }

    // Resize or clear other relevant vectors
    predicted_vals.clear();
    predictive_errors.clear();
    thresholds.clear();

    // Reset the threshold to the minimal threshold
    thresholds.push_back(minimal_threshold);

    // Train the generator if necessary
    if (observed_vals.size() >= predictor_config.lookback_len) {
        std::vector<float> training_data(observed_vals.end() - predictor_config.lookback_len, observed_vals.end());
        generator.train(config::epoch_train, config::lr_train, training_data);
    }

    std::cout << "Training data set with " << observed_vals.size() << " samples." << std::endl;
}

bool AdapAD::is_anomalous(float observed_val, bool actual_anomaly) {
    try {
        float normalized_val = normalize_data(observed_val);
        observed_vals.push_back(normalized_val);
        
        if (observed_vals.size() < predictor_config.lookback_len) {
            std::cout << "Not enough data yet. Returning false." << std::endl;
            return false;
        }

        // Prepare input for prediction
        std::vector<float> input = prepare_data_for_prediction(normalized_val);
        
        // Predict
        std::vector<float> predicted = data_predictor.predict(input);
        if (predicted.empty()) {
            std::cerr << "Error: Empty prediction from data_predictor" << std::endl;
            return false;
        }
        float predicted_val = predicted[0];
        predicted_vals.push_back(predicted_val);

        // Calculate error
        float error = std::abs(normalized_val - predicted_val);
        predictive_errors.push_back(error);

        // Generate threshold
        float threshold;
        if (predictive_errors.size() >= predictor_config.lookback_len) {
            std::vector<float> recent_errors(predictive_errors.end() - predictor_config.lookback_len, predictive_errors.end());
            threshold = generator.generate(recent_errors, minimal_threshold);
        } else {
            threshold = minimal_threshold;
        }
        
        // Determine if anomalous
        bool is_anomalous = (error > threshold);
        
        // Update generator if necessary
        if (is_anomalous || threshold > minimal_threshold) {
            if (predictive_errors.size() >= predictor_config.lookback_len) {
                std::vector<float> past_errors(predictive_errors.end() - predictor_config.lookback_len, predictive_errors.end());
                generator.update(config::update_G_lr, past_errors);
            }
        }
        
        // Log results
        log_results(is_anomalous, normalized_val, predicted_val, threshold, actual_anomaly);
        
        return is_anomalous;
    } catch (const std::exception& e) {
        std::cerr << "Error in is_anomalous: " << e.what() << std::endl;
        return false;
    }
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

float AdapAD::normalize_data(float val) const {
    return (val - value_range_config.lower_bound) / (value_range_config.upper_bound - value_range_config.lower_bound);
}

float AdapAD::reverse_normalized_data(float val) const {
    return val * (value_range_config.upper_bound - value_range_config.lower_bound) + value_range_config.lower_bound;
}

bool AdapAD::is_inside_range(float val) const {
    float observed_val = reverse_normalized_data(val);
    return observed_val >= value_range_config.lower_bound && observed_val <= value_range_config.upper_bound;
}

std::vector<float> AdapAD::prepare_data_for_prediction(float normalized_val) {
    std::vector<float> x_temp;
    if (observed_vals.size() >= predictor_config.lookback_len) {
        x_temp.assign(observed_vals.end() - predictor_config.lookback_len, observed_vals.end());
    } else {
        x_temp = observed_vals;
        x_temp.resize(predictor_config.lookback_len, normalized_val);  // Pad with the current value if not enough data
    }
    return x_temp;
}

void AdapAD::log_results(bool is_anomalous, float normalized_val, float predicted_val, float threshold, bool actual_anomaly) {
    if (f_log.is_open()) {
        float denormalized_val = reverse_normalized_data(normalized_val);
        float denormalized_predicted = reverse_normalized_data(predicted_val);
        f_log << denormalized_val << ","
              << denormalized_predicted << ","
              << denormalized_predicted - threshold << ","
              << denormalized_predicted + threshold << ","
              << (is_anomalous ? "true" : "false") << ","
              << (actual_anomaly ? "1" : "0") << ","
              << std::abs(normalized_val - predicted_val) << ","
              << threshold << std::endl;
    }
}

std::string AdapAD::get_log_filename() const {
    return config::log_file_path;
}
