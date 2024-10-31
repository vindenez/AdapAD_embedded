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
    // Store the normalized minimal threshold
    float normalized_minimal_threshold = minimal_threshold / (value_range_config.upper_bound - value_range_config.lower_bound);
    thresholds.push_back(normalized_minimal_threshold);
    std::cout << "Minimal threshold: " << minimal_threshold << " (normalized: " << normalized_minimal_threshold << ")" << std::endl;

    f_name = "adapad_log.csv";
    std::cout << "Log file: " << f_name << std::endl;
    open_log_file();
}

void AdapAD::set_training_data(const std::vector<float>& data) {
    observed_vals.clear();
    
    // Only take train_size points
    size_t train_size = predictor_config.train_size;
    auto training_window = std::vector<float>(data.end() - train_size, data.end());
    
    for (const auto& val : training_window) {
        observed_vals.push_back(normalize_data(val));
    }
    
    train(observed_vals);
}

void AdapAD::train(const std::vector<float>& data) {
    std::cout << "\n=== Initial Training Phase ===" << std::endl;
    // Train predictor
    auto [trainX, trainY] = data_predictor.train(config::epoch_train, config::lr_train, data);
    std::cout << "✓ Trained NormalDataPredictor" << std::endl;

    // Learn error patterns
    predicted_vals.clear();
    for (const auto& x : trainX) {
        auto train_predicted_val = data_predictor.predict(x);
        predicted_vals.push_back(train_predicted_val[0]);
    }

    std::vector<float> observed_vals_;
    for (const auto& y : trainY) {
        observed_vals_.insert(observed_vals_.end(), y.begin(), y.end());
    }

    auto predictive_errors = calc_error(observed_vals_, predicted_vals);
    
    // Generator warmup training
    std::cout << "\n=== Generator Warmup Training ===" << std::endl;
    generator.train(config::epoch_train, config::lr_train, predictive_errors);
    
    // Initialize thresholds with minimal_threshold
    float normalized_minimal_threshold = minimal_threshold / (value_range_config.upper_bound - value_range_config.lower_bound);
    thresholds = std::vector<float>(predictive_errors.size(), normalized_minimal_threshold);
    
    std::cout << "✓ Trained AnomalousThresholdGenerator\n" << std::endl;
}

bool AdapAD::is_anomalous(float observed_val, bool actual_anomaly) {
    bool is_anomalous_ret = false;
    float normalized_threshold = minimal_threshold / (value_range_config.upper_bound - value_range_config.lower_bound);
    std::vector<float> predicted_val_vec;
    
    // Normalize the observed value
    float normalized_observed = normalize_data(observed_val);
    observed_vals.push_back(normalized_observed);
    size_t supposed_anomalous_pos = observed_vals.size();

    // First check: range violation
    if (!is_inside_range(normalized_observed)) {
        is_anomalous_ret = true;
        anomalies.push_back(supposed_anomalous_pos);
    } else {
        auto past_observations = prepare_data_for_prediction();
        predicted_val_vec = data_predictor.predict(past_observations);
        
        if (!predicted_val_vec.empty()) {
            float predicted_val = predicted_val_vec[0];
            predicted_vals.push_back(predicted_val);

            // Calculate error in normalized space
            float prediction_error = std::abs(normalized_observed - predicted_val);
            predictive_errors.push_back(prediction_error);
            
            if (predictive_errors.size() >= predictor_config.lookback_len) {
                std::vector<float> past_predictive_errors(predictive_errors.end() - predictor_config.lookback_len, predictive_errors.end());
                
                // Get threshold in normalized space
                normalized_threshold = generator.generate(past_predictive_errors, normalized_threshold);
                normalized_threshold = std::max(normalized_threshold, minimal_threshold / (value_range_config.upper_bound - value_range_config.lower_bound));
                thresholds.push_back(normalized_threshold);

                // Calculate bounds and check if value is outside them
                float lower_bound = predicted_val - normalized_threshold;
                float upper_bound = predicted_val + normalized_threshold;
                
                // Compare in normalized space
                if ((normalized_observed < lower_bound || normalized_observed > upper_bound) && !is_default_normal()) {
                    is_anomalous_ret = true;
                    anomalies.push_back(supposed_anomalous_pos);
                }

                // Online learning updates
                if (!is_anomalous_ret) {
                    std::vector<float> recent_observation{normalized_observed};
                    data_predictor.update(config::epoch_update, config::lr_update, past_observations, recent_observation);
                    
                    if (normalized_threshold > minimal_threshold / (value_range_config.upper_bound - value_range_config.lower_bound)) {
                        update_generator(past_predictive_errors, prediction_error);
                    }
                }
            }
        }
    }

    log_result(is_anomalous_ret, normalized_observed, predicted_val_vec.empty() ? 0 : predicted_val_vec[0], 
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
    std::vector<float> x_temp(observed_vals.end() - predictor_config.lookback_len - 1, observed_vals.end() - 1);
    auto predicted_vals_tail = std::vector<float>(predicted_vals.end() - predictor_config.lookback_len, predicted_vals.end());

    // Replace out-of-range values with predictions
    for (int i = x_temp.size() - 1; i >= 0; --i) {
        if (!is_inside_range(x_temp[i])) {
            size_t pred_idx = predicted_vals_tail.size() - x_temp.size() + i;
            if (pred_idx < predicted_vals_tail.size()) {
                x_temp[i] = predicted_vals_tail[pred_idx];
            }
        }
    }

    return x_temp;
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

void AdapAD::log_result(bool is_anomalous, float normalized_val, float predicted_val, float normalized_threshold, bool actual_anomaly) {
    if (f_log.is_open()) {
        float normalized_lower_bound = predicted_val - normalized_threshold;
        float normalized_upper_bound = predicted_val + normalized_threshold;
        
        normalized_lower_bound = std::max(0.0f, normalized_lower_bound);
        normalized_upper_bound = std::min(1.0f, normalized_upper_bound);
        
        float denormalized_val = reverse_normalized_data(normalized_val);
        float denormalized_predicted = reverse_normalized_data(predicted_val);
        float denormalized_lower = reverse_normalized_data(normalized_lower_bound);
        float denormalized_upper = reverse_normalized_data(normalized_upper_bound);
        float denormalized_threshold = normalized_threshold * (value_range_config.upper_bound - value_range_config.lower_bound);
        denormalized_threshold = std::max(denormalized_threshold, minimal_threshold);
        
        float error = std::abs(normalized_val - predicted_val);

        f_log << std::fixed << std::setprecision(6)
              << denormalized_val << ","
              << denormalized_predicted << ","
              << denormalized_lower << ","
              << denormalized_upper << ","
              << (is_anomalous ? "1" : "0") << ","
              << (actual_anomaly ? "1" : "0") << ","
              << error << ","
              << denormalized_threshold << std::endl;
    }
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
