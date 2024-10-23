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
    observed_vals.clear();
    for (const auto& val : data) {
        observed_vals.push_back(normalize_data(val));
    }
    
    train(observed_vals);
}

void AdapAD::train(const std::vector<float>& data) {
    auto [trainX, trainY] = data_predictor.train(config::epoch_train, config::lr_train, data);
    std::cout << "Trained NormalDataPredictor" << std::endl;

    for (const auto& x : trainX) {
        auto train_predicted_val = data_predictor.predict(x);
        predicted_vals.push_back(train_predicted_val[0]);
    }

    std::vector<float> observed_vals_;
    for (const auto& y : trainY) {
        observed_vals_.insert(observed_vals_.end(), y.begin(), y.end());
    }
    
    auto predictive_errors = calc_error(observed_vals_, 
        std::vector<float>(predicted_vals.end() - observed_vals_.size(), predicted_vals.end()));
    
    generator.train(config::epoch_train, config::lr_train, predictive_errors);

    auto predicted_vals_tail = std::vector<float>(predicted_vals.end() - observed_vals_.size(), predicted_vals.end());
    for (size_t i = 0; i < observed_vals_.size(); ++i) {
        log_result(false, observed_vals_[i], predicted_vals_tail[i], 0, false);
    }
    std::cout << "Trained AnomalousThresholdGenerator" << std::endl;
}

bool AdapAD::is_anomalous(float observed_val, bool actual_anomaly) {
    bool is_anomalous_ret = false;
    
    observed_val = normalize_data(observed_val);
    observed_vals.push_back(observed_val);
    size_t supposed_anomalous_pos = observed_vals.size();

    auto past_observations = prepare_data_for_prediction();
    auto predicted_val = data_predictor.predict(past_observations)[0];
    predicted_vals.push_back(predicted_val);

    if (!is_inside_range(observed_val)) {
        anomalies.push_back(supposed_anomalous_pos);
        is_anomalous_ret = true;
    } else {
        generator.eval();
        std::vector<float> past_predictive_errors(predictive_errors.end() - predictor_config.lookback_len, predictive_errors.end());
        float threshold = generator.generate(past_predictive_errors, minimal_threshold);
        thresholds.push_back(threshold);

        float prediction_error = std::abs(predicted_val - observed_val);
        predictive_errors.push_back(prediction_error);

        if (prediction_error > threshold) {
            if (!is_default_normal()) {
                is_anomalous_ret = true;
                anomalies.push_back(supposed_anomalous_pos);
            }
        }

        data_predictor.update(config::epoch_update, config::lr_update, past_observations, {observed_val});

        if (is_anomalous_ret || threshold > minimal_threshold) {
            update_generator(past_predictive_errors, prediction_error);
        }
    }

    log_result(is_anomalous_ret, observed_val, predicted_val, thresholds.back(), actual_anomaly);
    return is_anomalous_ret;
}

void AdapAD::clean() {
    if (predicted_vals.size() > predictor_config.lookback_len) {
        predicted_vals.erase(predicted_vals.begin(), predicted_vals.end() - predictor_config.lookback_len);
    }
    if (predictive_errors.size() > predictor_config.lookback_len) {
        predictive_errors.erase(predictive_errors.begin(), predictive_errors.end() - predictor_config.lookback_len);
    }
    if (thresholds.size() > predictor_config.lookback_len) {
        thresholds.erase(thresholds.begin(), thresholds.end() - predictor_config.lookback_len);
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
    std::cout << "Generator update average loss: " << avg_loss << std::endl;
}

void AdapAD::log_result(bool is_anomalous, float normalized_val, float predicted_val, float threshold, bool actual_anomaly) {
    if (f_log.is_open()) {
        float denormalized_val = reverse_normalized_data(normalized_val);
        float denormalized_predicted = reverse_normalized_data(predicted_val);
        float lower_bound = denormalized_predicted - threshold;
        float upper_bound = denormalized_predicted + threshold;
        float error = std::abs(normalized_val - predicted_val);
        
        f_log << std::fixed << std::setprecision(3)
              << denormalized_val << ","                  // Observed
              << denormalized_predicted << ","            // Predicted
              << lower_bound << ","                       // LowerBound
              << upper_bound << ","                       // UpperBound
              << (is_anomalous ? "1" : "0") << ","        // PredictedAnomaly
              << (actual_anomaly ? "1" : "0") << ","      // ActualAnomaly
              << error << ","                             // Error
              << threshold << std::endl;                  // Threshold
    }
}

void AdapAD::open_log_file() {
    bool file_exists = std::filesystem::exists(f_name);

    f_log.open(f_name, std::ios::app);
    if (!f_log.is_open()) {
        std::cerr << "Error: Could not open log file: " << f_name << std::endl;
        std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
        throw std::runtime_error("Could not open log file: " + f_name);
    }

    if (!file_exists) {
        f_log << "Observed,Predicted,LowerBound,UpperBound,PredictedAnomaly,ActualAnomaly,Error,Threshold\n";
    }
}

std::vector<float> AdapAD::calc_error(const std::vector<float>& ground_truth, const std::vector<float>& predict) {
    if (ground_truth.size() != predict.size()) {
        throw std::invalid_argument("Ground truth and prediction vectors must have the same size");
    }

    std::vector<float> errors;
    errors.reserve(ground_truth.size());

    for (size_t i = 0; i < ground_truth.size(); ++i) {
        errors.push_back(pow(ground_truth[i] - predict[i], 2));
    }

    return errors;
}
