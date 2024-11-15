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
    : data_predictor(predictor_config.num_layers,
                    predictor_config.hidden_size,
                    predictor_config.lookback_len,
                    predictor_config.prediction_len),
      generator(predictor_config.num_layers,
               predictor_config.hidden_size,
               predictor_config.lookback_len,
               predictor_config.prediction_len),
      predictor_config(predictor_config),
      sensor_range(),
      minimal_threshold(minimal_threshold)
{
    sensor_range.init(value_range_config.lower_bound, value_range_config.upper_bound);
    
    thresholds.push_back(minimal_threshold);
    
    f_name = config::log_file_path;
    std::cout << "Log file: " << f_name << std::endl;
    open_log_file();
}


void AdapAD::set_training_data(const std::vector<float>& data) {
    std::cout << "\n=== Training Data Range Analysis ===" << std::endl;
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    std::cout << "Actual data range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "Configured sensor range: [" << sensor_range.lower() << ", " 
              << sensor_range.upper() << "]" << std::endl;
    
    observed_vals.clear();
    size_t train_size = predictor_config.train_size;
    auto training_window = std::vector<float>(data.end() - train_size, data.end());
    
    // Process all values - anomalies will normalize to values outside [0,1]
    for (const auto& val : training_window) {
        float normalized_val = normalize_data(val);
        observed_vals.push_back(normalized_val);
        std::cout << "Original: " << val << " -> Normalized: " << normalized_val << std::endl;
    }

    // Add verification
    float min_normalized = *std::min_element(observed_vals.begin(), observed_vals.end());
    float max_normalized = *std::max_element(observed_vals.begin(), observed_vals.end());
    std::cout << "Normalized range: [" << min_normalized << ", " << max_normalized << "]" << std::endl;
    
    // Verify denormalization
    float test_val = reverse_normalized_data(0.0f);
    std::cout << "Denormalization test - 0.0 maps to: " << test_val << std::endl;
}

void AdapAD::train(float measured_value) {
    std::cout << "train method called" << std::endl;
    std::cout << "\n=== Initial Training Phase ===" << std::endl;
    
    auto [trainX, trainY] = data_predictor.train(predictor_config.epoch_train, 
                                              predictor_config.lr_train, 
                                              observed_vals);
    
    predicted_vals.clear();
    predictive_errors.clear();
    
    for (size_t i = 0; i < trainX.size(); i++) {
        float predicted_val = data_predictor.predict(trainX[i]);
        predicted_vals.push_back(predicted_val);
        
        float error = std::abs(trainY[i][0] - predicted_val);
        error = std::min(error, 0.2f);
        predictive_errors.push_back(error);
    }
    
    generator.train(predictor_config.epoch_train,
                   predictor_config.lr_train,
                   predictive_errors);
    std::cout << "Trained AnomalousThresholdGenerator" << std::endl;
}

bool AdapAD::is_anomalous(float val, bool actual_anomaly) {
    float normalized_observed = normalize_data(val);
    
    bool is_anomalous_ret = false;
    float predicted_val = 0.0f;
    float threshold = minimal_threshold;

    std::cout << "\n=== Anomaly Detection Debug ===" << std::endl;
    std::cout << "Raw measured value: " << val << std::endl;

    
    observed_vals.push_back(normalized_observed);
    
    if (!is_inside_range(normalized_observed)) {
        anomalies.push_back(observed_vals.size());
        is_anomalous_ret = true;
    } else if (observed_vals.size() >= predictor_config.lookback_len + 1) {
        // Get past observations for prediction
        std::vector<float> past_observations = prepare_data_for_prediction();
        
        // Debug prediction inputs
        std::cout << "Prediction input sequence:" << std::endl;
        for (float f : past_observations) {
            std::cout << reverse_normalized_data(f) << " ";
        }
        std::cout << std::endl;
        
        predicted_val = data_predictor.predict(past_observations);
        predicted_vals.push_back(predicted_val);
        std::cout << "Normalized prediction: " << predicted_val << std::endl;

        
        // Calculate prediction error and threshold
        float prediction_error = std::abs(normalized_observed - predicted_val);
        prediction_error = std::min(prediction_error, 0.2f);  // Cap error like Python
        
        // Generate threshold
        std::vector<float> past_errors(predictive_errors.end() - predictor_config.lookback_len, 
                                     predictive_errors.end());
        threshold = generator.generate(past_errors, minimal_threshold);
        thresholds.push_back(threshold);
        
        // Only mark as anomaly if error exceeds threshold AND not in default normal state
        if (prediction_error > threshold && !is_default_normal()) {
            is_anomalous_ret = true;
            anomalies.push_back(observed_vals.size());
        }
        
        predictive_errors.push_back(prediction_error);
        
        // Update models
        data_predictor.update(config::epoch_update, config::lr_update, 
                            past_observations, normalized_observed);
        
        if (is_anomalous_ret || threshold > minimal_threshold) {
            update_generator(past_errors, prediction_error);
        }
    }
    
    log_result(is_anomalous_ret, normalized_observed, predicted_val, 
               threshold, actual_anomaly);
    
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
    float range = sensor_range.upper() - sensor_range.lower();
    float normalized = (val - sensor_range.lower()) / range;
    
    return normalized;
}

float AdapAD::reverse_normalized_data(float val) const {
    return val * (sensor_range.upper() - sensor_range.lower()) + sensor_range.lower();
}

bool AdapAD::is_inside_range(float normalized_val) const {
    float denormalized = reverse_normalized_data(normalized_val);
    return denormalized >= sensor_range.lower() && denormalized <= sensor_range.upper();
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
    generator.update(config::update_G_epoch, config::update_G_lr, past_observations, observed_val);
}

void AdapAD::log_result(bool is_anomalous, float observed, float predicted, float threshold, bool actual_anomaly) {
    std::cout << "\n=== Logging Debug ===" << std::endl;
    std::cout << "Normalized values:" << std::endl;
    std::cout << "  Observed: " << observed << std::endl;
    std::cout << "  Predicted: " << predicted << std::endl;
    std::cout << "  Threshold: " << threshold << std::endl;
    std::cout << "Denormalized values:" << std::endl;
    std::cout << "  Observed: " << reverse_normalized_data(observed) << std::endl;
    std::cout << "  Predicted: " << reverse_normalized_data(predicted) << std::endl;
    std::cout << "  Lower bound: " << reverse_normalized_data(predicted - threshold) << std::endl;
    std::cout << "  Upper bound: " << reverse_normalized_data(predicted + threshold) << std::endl;
    
    // Get the latest values
    float latest_observed = observed_vals.back();
    float latest_predicted = predicted_vals.back();
    float latest_threshold = thresholds.back();
    float latest_error = predictive_errors.back();

    // Calculate bounds in normalized space (like Python)
    float normalized_lower = latest_predicted - latest_threshold;
    float normalized_upper = latest_predicted + latest_threshold;

    f_log.open(f_name, std::ios::app);
    f_log << reverse_normalized_data(latest_observed) << ","
          << reverse_normalized_data(latest_predicted) << ","
          << reverse_normalized_data(normalized_lower) << ","
          << reverse_normalized_data(normalized_upper) << ","
          << (is_anomalous ? "1" : "0") << ","
          << (actual_anomaly ? "1" : "0") << ","
          << latest_error << ","
          << latest_threshold << "\n";
    f_log.flush();
    f_log.close();
}

void AdapAD::open_log_file() {
    f_log.open(f_name, std::ios::out | std::ios::trunc);
    if (!f_log.is_open()) {
        std::cerr << "Error: Could not open log file: " << f_name << std::endl;
        std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
        throw std::runtime_error("Could not open log file: " + f_name);
    }

    // Include actual_anomaly in header
    f_log << "observed,predicted,low,high,anomalous,actual_anomaly,err,threshold\n";
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
    
    // Clear all vectors before destruction
    observed_vals.clear();
    predicted_vals.clear();
    predictive_errors.clear();
    thresholds.clear();
    anomalies.clear();
    
    // Close file if open
    if (f_log.is_open()) {
        f_log.close();
    }
}
