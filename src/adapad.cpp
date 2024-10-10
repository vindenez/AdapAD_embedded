#include "adapad.hpp"
#include "json_loader.hpp"
#include "config.hpp"  // Include config to access global variables like update_G_epoch and update_G_lr
#include <iostream>
#include <cmath>
#include "config.hpp"

AdapAD::AdapAD(const PredictorConfig& predictor_config, 
               const ValueRangeConfig& value_range_config, 
               float minimal_threshold,
               const LSTMPredictor& lstm_predictor)
    : data_predictor(create_normal_data_predictor(lstm_predictor)),
      generator(predictor_config.lookback_len, predictor_config.prediction_len, 
                value_range_config.lower_bound, value_range_config.upper_bound),
      minimal_threshold(minimal_threshold),
      predictor_config(predictor_config),
      value_range_config(value_range_config) {
    
    try {
        observed_vals.reserve(predictor_config.lookback_len);
        predicted_vals.reserve(predictor_config.lookback_len);
        predictive_errors.reserve(predictor_config.lookback_len);
        thresholds.reserve(predictor_config.lookback_len);
        thresholds.push_back(minimal_threshold);

        // Log file setup
        f_log.open("anomalous_detection_log.csv");
        if (!f_log.is_open()) {
            throw std::runtime_error("Could not open log file.");
        }
        f_log << "Observed,Predicted,LowerBound,UpperBound,Anomalous,Error,Threshold\n";
    } catch (const std::exception& e) {
        std::cerr << "Error in AdapAD constructor: " << e.what() << std::endl;
        throw;
    }
}

// Helper function to create NormalDataPredictor from LSTMPredictor
NormalDataPredictor AdapAD::create_normal_data_predictor(const LSTMPredictor& lstm_predictor) {
    try {
        std::unordered_map<std::string, std::vector<std::vector<float>>> weights;
        std::unordered_map<std::string, std::vector<float>> biases;

        // Assuming LSTMPredictor has getter methods for weights and biases
        weights["lstm.weight_ih_l0"] = lstm_predictor.get_weight_ih_input();
        weights["lstm.weight_hh_l0"] = lstm_predictor.get_weight_hh_input();
        biases["lstm.bias_ih_l0"] = lstm_predictor.get_bias_ih_input();
        biases["lstm.bias_hh_l0"] = lstm_predictor.get_bias_hh_input();

        // Add FC layer weights and biases
        weights["fc.weight"] = std::vector<std::vector<float>>(1, std::vector<float>(lstm_predictor.get_hidden_size(), 0.1f));
        biases["fc.bias"] = std::vector<float>(1, 0.0f);

        if (weights["lstm.weight_ih_l0"].empty() || weights["lstm.weight_hh_l0"].empty() ||
            biases["lstm.bias_ih_l0"].empty() || biases["lstm.bias_hh_l0"].empty()) {
            throw std::runtime_error("Empty LSTM weights or biases");
        }

        std::cout << "Creating NormalDataPredictor with:" << std::endl;
        std::cout << "weight_ih_l0 dimensions: " << weights["lstm.weight_ih_l0"].size() << " x " << (weights["lstm.weight_ih_l0"].empty() ? 0 : weights["lstm.weight_ih_l0"][0].size()) << std::endl;
        std::cout << "weight_hh_l0 dimensions: " << weights["lstm.weight_hh_l0"].size() << " x " << (weights["lstm.weight_hh_l0"].empty() ? 0 : weights["lstm.weight_hh_l0"][0].size()) << std::endl;
        std::cout << "bias_ih_l0 size: " << biases["lstm.bias_ih_l0"].size() << std::endl;
        std::cout << "bias_hh_l0 size: " << biases["lstm.bias_hh_l0"].size() << std::endl;
        std::cout << "fc.weight dimensions: " << weights["fc.weight"].size() << " x " << (weights["fc.weight"].empty() ? 0 : weights["fc.weight"][0].size()) << std::endl;
        std::cout << "fc.bias size: " << biases["fc.bias"].size() << std::endl;

        return NormalDataPredictor(weights, biases);
    } catch (const std::exception& e) {
        std::cerr << "Error creating NormalDataPredictor: " << e.what() << std::endl;
        throw;
    }
}

void AdapAD::set_training_data(const std::vector<float>& data) {
    observed_vals = data;
}

bool AdapAD::is_anomalous(float observed_val) {
    try {
        observed_vals.push_back(observed_val);

        if (observed_vals.size() < predictor_config.lookback_len) {
            std::cerr << "Not enough data to perform prediction. Need at least " 
                      << predictor_config.lookback_len << " values." << std::endl;
            return false;
        }

        // Prepare input for prediction
        std::vector<float> input(observed_vals.end() - predictor_config.lookback_len, observed_vals.end());

        // Predict
        float predicted_val = data_predictor.predict(input)[0];
        predicted_vals.push_back(predicted_val);

        // Calculate error
        float error = std::abs(observed_val - predicted_val);
        predictive_errors.push_back(error);

        // Generate threshold
        std::vector<float> past_errors(predictive_errors.end() - predictor_config.lookback_len, predictive_errors.end());
        float threshold = generator.generate(past_errors, minimal_threshold);
        thresholds.push_back(threshold);

        // Determine if anomalous
        bool is_anomalous = (error > threshold);

        // Update predictor
        data_predictor.update(predictor_config.epoch_update, predictor_config.lr_update, input, observed_val);

        // Update generator if necessary
        if (is_anomalous || threshold > minimal_threshold) {
            generator.update(config::update_G_epoch, config::update_G_lr, past_errors, error);
        }

        // Log the result
        if (f_log.is_open()) {
            f_log << observed_val << "," << predicted_val << ","
                  << predicted_val - threshold << "," << predicted_val + threshold << ","
                  << is_anomalous << "," << error << "," << threshold << "\n";
        }

        // Clean up old data if necessary
        clean();

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