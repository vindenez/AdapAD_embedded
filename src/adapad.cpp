#include "adapad.hpp"
#include "json_loader.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <filesystem>

AdapAD::AdapAD(const PredictorConfig& predictor_config,
               const ValueRangeConfig& value_range_config,
               float minimal_threshold)
    : predictor_config(predictor_config),
      value_range_config(value_range_config),
      minimal_threshold(minimal_threshold),
      data_predictor(predictor_config.num_layers,
                     predictor_config.hidden_size,
                     predictor_config.lookback_len,
                     predictor_config.prediction_len),
      generator(predictor_config.num_layers,
                predictor_config.hidden_size,
                predictor_config.lookback_len,
                predictor_config.prediction_len)
{
    thresholds.push_back(minimal_threshold);
    std::cout << "Minimal threshold: " << minimal_threshold << std::endl;

    f_name = "adapad_log.csv";
    std::cout << "Log file: " << f_name << std::endl;
    open_log_file();
}

void AdapAD::set_training_data(const std::vector<float>& data) {
    std::cout << "set_training_data called with " << data.size() << " elements" << std::endl;
    observed_vals.clear();
    
    // Only take train_size points
    size_t train_size = predictor_config.train_size;
    auto training_window = std::vector<float>(data.end() - train_size, data.end());
    
    for (const auto& val : training_window) {
        observed_vals.push_back(normalize_data(val));
    }
    
    std::cout << "observed_vals size after set_training_data: " << observed_vals.size() << std::endl;
}

void AdapAD::train(const std::vector<float>& data) {
    std::cout << "train method called" << std::endl;
    std::cout << "\n=== Initial Training Phase ===" << std::endl;
    
    // Train the normal data predictor
    auto [trainX, trainY] = data_predictor.train(predictor_config.epoch_train, 
                                              predictor_config.lr_train, 
                                              data,
                                              nullptr);  
    
    // Store predictions for training data
    predicted_vals.clear();
    predictive_errors.clear();
    
    for (size_t i = 0; i < trainX.size(); i++) {
        float predicted_val = data_predictor.predict(trainX[i]);
        predicted_vals.push_back(predicted_val);
        float error = std::abs(trainY[i][0] - predicted_val);
        error = std::min(error, 0.2f);  // Cap error like in Python
        predictive_errors.push_back(error);
    }
    
    // Train the generator using prediction errors
    generator.train(predictor_config.epoch_train,
                   predictor_config.lr_train,
                   predictive_errors);
    std::cout << "Trained AnomalousThresholdGenerator" << std::endl;
}

bool AdapAD::is_anomalous(float val, bool actual_anomaly) {
    bool is_anomalous_ret = false;
    float predicted_val = 0.0f;
    float normalized_threshold = minimal_threshold;
    
    // Normalize and save observed value (like Python)
    float normalized_observed = normalize_data(val);
    observed_vals.push_back(normalized_observed);
    size_t supposed_anomalous_pos = observed_vals.size();  // Match Python's get_length()
    
    // First perform range check (like Python)
    if (!is_inside_range(normalized_observed)) {
        anomalies.push_back(supposed_anomalous_pos);
        is_anomalous_ret = true;
    } else if (observed_vals.size() >= predictor_config.lookback_len + 1) {
        // Get past observations for prediction
        std::vector<float> past_observations = prepare_data_for_prediction();
        
        // Make prediction
        predicted_val = data_predictor.predict(past_observations);
        predicted_vals.push_back(predicted_val);
        
        // Calculate error
        float prediction_error = std::abs(normalized_observed - predicted_val);
        prediction_error = std::min(prediction_error, 0.2f);  // Cap error like Python
        predictive_errors.push_back(prediction_error);
        
        if (predictive_errors.size() >= predictor_config.lookback_len) {
            // Get past errors for threshold generation
            std::vector<float> past_errors(
                predictive_errors.end() - predictor_config.lookback_len,
                predictive_errors.end()
            );
            
            // Generate threshold and apply minimal threshold directly (like Python)
            normalized_threshold = generator.generate(past_errors, minimal_threshold);
            normalized_threshold = std::max(normalized_threshold, minimal_threshold);
            thresholds.push_back(normalized_threshold);
            
            // Check for anomaly
            if (prediction_error > normalized_threshold) {
                if (!is_default_normal()) {
                    is_anomalous_ret = true;
                    anomalies.push_back(supposed_anomalous_pos);
                }
            }
            
            // Update models
            data_predictor.update(predictor_config.epoch_update,
                                predictor_config.lr_update,
                                past_observations,
                                normalized_observed);
            
            if (is_anomalous_ret || normalized_threshold > minimal_threshold) {
                generator.update(predictor_config.epoch_update,
                               predictor_config.lr_update,
                               past_errors,
                               prediction_error);
            }
        }
    }
    
    log_result(is_anomalous_ret, normalized_observed, predicted_val,
               normalized_threshold, actual_anomaly);
    
    return is_anomalous_ret;
}

void AdapAD::clean() {
    size_t keep_size = std::max(predictor_config.lookback_len * 2, static_cast<int>(observed_vals.size() * 0.1));
    if (observed_vals.size() > keep_size) {
        observed_vals.erase(observed_vals.begin(), observed_vals.end() - keep_size);
    }
    if (predicted_vals.size() > keep_size) {
        predicted_vals.erase(predicted_vals.begin(), predicted_vals.end() - keep_size);
    }
    if (predictive_errors.size() > keep_size) {
        predictive_errors.erase(predictive_errors.begin(), predictive_errors.end() - keep_size);
    }
    if (thresholds.size() > keep_size) {
        thresholds.erase(thresholds.begin(), thresholds.end() - keep_size);
    }
}

float AdapAD::normalize_data(float val) const {
    return (val - value_range_config.lower_bound) / 
           (value_range_config.upper_bound - value_range_config.lower_bound);
}

float AdapAD::reverse_normalized_data(float val) const {
    return val * (value_range_config.upper_bound - value_range_config.lower_bound) + 
           value_range_config.lower_bound;
}

bool AdapAD::is_inside_range(float val) const {
    float observed_val = reverse_normalized_data(val);
    return observed_val >= value_range_config.lower_bound && observed_val <= value_range_config.upper_bound;
}

std::vector<float> AdapAD::prepare_data_for_prediction() {
    // Get lookback window excluding the current value (like Python's [:-1])
    auto _x_temp = std::vector<float>(
        observed_vals.end() - (predictor_config.lookback_len + 1),
        observed_vals.end() - 1
    );
    
    // Get recent predictions (like Python's predicted_vals.get_tail())
    auto predicted_vals_tail = std::vector<float>(
        predicted_vals.end() - predictor_config.lookback_len,
        predicted_vals.end()
    );
    
    // Replace out-of-range values with predictions (matching Python indexing)
    for (int i = 0; i < predictor_config.lookback_len; i++) {
        int checked_pos = i + 1;  // Match Python's indexing
        
        // Convert positive index to negative index for _x_temp
        int pos_from_end = _x_temp.size() - checked_pos;
        
        if (!is_inside_range(_x_temp[pos_from_end])) {
            _x_temp[pos_from_end] = predicted_vals_tail[predicted_vals_tail.size() - checked_pos];
        }
    }
    
    return _x_temp;
}

bool AdapAD::is_default_normal() const {
    auto observed_vals_tail = std::vector<float>(observed_vals.end() - predictor_config.train_size, observed_vals.end());
    int cnt = 0;
    for (const auto& val : observed_vals_tail) {
        if (!is_inside_range(val)) {
            cnt++;
        }
    }
    return cnt > predictor_config.train_size / 2;
}

void AdapAD::update_generator(const std::vector<float>& past_observations, float observed_val) {
    generator.train();
    float avg_loss = generator.update(config::update_G_epoch, config::update_G_lr, past_observations, observed_val);
}

void AdapAD::log_result(bool is_anomalous, float observed, float predicted, float threshold, bool actual_anomaly) {
    // Calculate bounds in normalized space
    float normalized_lower = predicted - threshold;
    float normalized_upper = predicted + threshold;
    
    // Then denormalize everything
    float denorm_observed = reverse_normalized_data(observed);
    float denorm_predicted = reverse_normalized_data(predicted);
    float denorm_lower = reverse_normalized_data(normalized_lower);
    float denorm_upper = reverse_normalized_data(normalized_upper);
    
    f_log << denorm_observed << ","
          << denorm_predicted << ","
          << denorm_lower << ","
          << denorm_upper << ","
          << (is_anomalous ? "1" : "0") << ","
          << (actual_anomaly ? "1" : "0") << ","
          << std::abs(observed - predicted) << "," // Keep error normalized
          << threshold << "\n";  // Keep threshold normalized
    f_log.flush();
}

void AdapAD::open_log_file() {
    f_log.open(f_name, std::ios::out | std::ios::trunc);
    if (!f_log.is_open()) {
        std::cerr << "Error: Could not open log file: " << f_name << std::endl;
        std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
        throw std::runtime_error("Could not open log file: " + f_name);
    }

    f_log << "Observed,Predicted,LowerBound,UpperBound,PredictedAnomaly,ActualAnomaly,Error,Threshold\n";
}

std::vector<float> AdapAD::calc_error(const std::vector<float>& ground_truth, const std::vector<float>& predict) {
    std::vector<float> errors;
    errors.reserve(ground_truth.size());

    for (size_t i = 0; i < ground_truth.size(); ++i) {
        errors.push_back(std::abs(ground_truth[i] - predict[i]));
    }

    return errors;
}

void AdapAD::maintain_memory() {
    size_t keep_size = std::max(predictor_config.lookback_len * 2, static_cast<int>(observed_vals.size() * 0.1));
    if (observed_vals.size() > keep_size) {
        observed_vals.erase(observed_vals.begin(), observed_vals.end() - keep_size);
    }
    if (predicted_vals.size() > keep_size) {
        predicted_vals.erase(predicted_vals.begin(), predicted_vals.end() - keep_size);
    }
    if (predictive_errors.size() > keep_size) {
        predictive_errors.erase(predictive_errors.begin(), predictive_errors.end() - keep_size);
    }
    if (thresholds.size() > keep_size) {
        thresholds.erase(thresholds.begin(), thresholds.end() - keep_size);
    }
}

AdapAD::~AdapAD() {
    std::cout << "AdapAD destructor called" << std::endl;
}
