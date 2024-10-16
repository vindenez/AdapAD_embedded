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

        // Train the generator
        std::vector<float> normalized_training_data;
        for (const auto& val : training_data) {
            normalized_training_data.push_back(normalize_data(val));
        }
        if (!normalized_training_data.empty()) {
            generator.train(config::epoch_train, config::lr_train, normalized_training_data);
        }
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

bool AdapAD::is_anomalous(float observed_val, bool actual_anomaly, bool log_results) {
    std::cout << "is_anomalous called with observed_val: " << observed_val 
              << ", actual_anomaly: " << (actual_anomaly ? "true" : "false") << std::endl;
    
    try {
        float normalized_val = normalize_data(observed_val);
        observed_vals.push_back(normalized_val);
        
        // Prepare data for prediction
        std::vector<float> past_observations = prepare_data_for_prediction();
        
        // Predict normal value using the pre-trained NormalDataPredictor
        std::vector<float> predicted = data_predictor.predict(past_observations);
        if (predicted.empty()) {
            throw std::runtime_error("Empty prediction from data_predictor");
        }
        float predicted_val = predicted[0];
        predicted_vals.push_back(predicted_val);
        
        // Calculate prediction error
        float prediction_error = std::abs(predicted_val - normalized_val);
        predictive_errors.push_back(prediction_error);
        
        // Generate threshold using the AnomalousThresholdGenerator
        std::vector<float> past_predictive_errors(predictive_errors.end() - predictor_config.lookback_len, predictive_errors.end());
        float threshold = generator.generate(past_predictive_errors, minimal_threshold);
        thresholds.push_back(threshold);
        
        // Determine if the point is anomalous
        bool is_anomalous_ret = prediction_error > threshold;
        
        // Log results if required
        if (log_results) {
            log_result(is_anomalous_ret, normalized_val, predicted_val, threshold, actual_anomaly);
        }
        
        return is_anomalous_ret;
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

std::vector<float> AdapAD::prepare_data_for_prediction() {
    std::vector<float> x_temp;
    
    // Get the last lookback_len + 1 observed values
    size_t start_idx = (observed_vals.size() > predictor_config.lookback_len + 1) 
                       ? observed_vals.size() - predictor_config.lookback_len - 1 
                       : 0;
    x_temp.assign(observed_vals.begin() + start_idx, observed_vals.end() - 1);

    // Get the last lookback_len predicted values
    std::vector<float> predicted_vals_tail;
    size_t pred_start_idx = (predicted_vals.size() > predictor_config.lookback_len) 
                            ? predicted_vals.size() - predictor_config.lookback_len 
                            : 0;
    predicted_vals_tail.assign(predicted_vals.begin() + pred_start_idx, predicted_vals.end());

    // Replace out-of-range values with predicted values
    for (int i = x_temp.size() - 1; i >= 0; --i) {
        if (!is_inside_range(x_temp[i])) {
            size_t pred_idx = predicted_vals_tail.size() - x_temp.size() + i;
            if (pred_idx < predicted_vals_tail.size()) {
                x_temp[i] = predicted_vals_tail[pred_idx];
            }
        }
    }

    // Ensure x_temp has exactly lookback_len elements
    if (x_temp.size() > predictor_config.lookback_len) {
        x_temp.erase(x_temp.begin(), x_temp.end() - predictor_config.lookback_len);
    } else if (x_temp.size() < predictor_config.lookback_len) {
        x_temp.insert(x_temp.begin(), predictor_config.lookback_len - x_temp.size(), 0.0f);
    }

    return x_temp;
}

void AdapAD::log_result(bool is_anomalous, float normalized_val, float predicted_val, float threshold, bool actual_anomaly) {
    if (f_log.is_open()) {
        float denormalized_val = reverse_normalized_data(normalized_val);
        float denormalized_predicted = reverse_normalized_data(predicted_val);
        
        // Debug output
        std::cout << "Logging result: " 
                  << "Actual anomaly: " << (actual_anomaly ? "true" : "false")
                  << ", Predicted anomaly: " << (is_anomalous ? "true" : "false")
                  << ", Value: " << denormalized_val 
                  << std::endl;

        f_log << denormalized_val << ","
              << denormalized_predicted << ","
              << denormalized_predicted - threshold << ","
              << denormalized_predicted + threshold << ","
              << (is_anomalous ? "1" : "0") << ","
              << (actual_anomaly ? "1" : "0") << ","
              << std::abs(normalized_val - predicted_val) << ","
              << threshold << std::endl;
    }
}

std::string AdapAD::get_log_filename() const {
    return config::log_file_path;
}

void AdapAD::warmup_generator(const std::vector<float>& normalized_data) {
    std::cout << "Starting warmup_generator with data size: " << normalized_data.size() << std::endl;
    std::cout << "Expected lookback_len: " << predictor_config.lookback_len << std::endl;
    std::cout << "First few values of normalized_data: ";
    for (size_t i = 0; i < std::min(size_t(5), normalized_data.size()); ++i) {
        std::cout << normalized_data[i] << " ";
    }
    std::cout << std::endl;

    if (normalized_data.size() < predictor_config.lookback_len) {
        std::cerr << "Error: Insufficient data for generator warmup. Expected at least: " 
                  << predictor_config.lookback_len << ", Got: " << normalized_data.size() << std::endl;
        return;
    }
    
    // Prepare data for training
    std::vector<std::vector<float>> training_data;
    for (size_t i = 0; i <= normalized_data.size() - predictor_config.lookback_len; ++i) {
        std::vector<float> window(normalized_data.begin() + i, normalized_data.begin() + i + predictor_config.lookback_len);
        training_data.push_back(window);
    }
    
    std::cout << "Warming up the generator..." << std::endl;
    std::cout << "Using epoch_train: " << config::epoch_train << ", lr_train: " << config::lr_train << std::endl;
    
    // Train the generator with the prepared data
    for (const auto& window : training_data) {
        generator.train(config::epoch_train, config::lr_train, window);
    }
    
    std::cout << "Generator warmup complete." << std::endl;
}

void AdapAD::open_log_file() {
    f_log.open(config::log_file_path);
    if (!f_log.is_open()) {
        throw std::runtime_error("Could not open log file.");
    }
    f_log << "Observed,Predicted,LowerBound,UpperBound,PredictedAnomaly,ActualAnomaly,Error,Threshold\n";
}
