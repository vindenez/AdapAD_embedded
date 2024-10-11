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
        f_log.open(config::log_file_path);
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
        // Load weights and biases from JSON file
        std::string weights_file = "weights/lstm_weights.json";
        auto weights = load_all_weights(weights_file);
        auto biases = load_all_biases(weights_file);

        if (weights.empty() || biases.empty()) {
            throw std::runtime_error("Failed to load weights and biases from JSON file");
        }

        // Use the hidden size from the config
        int hidden_size = config::LSTM_size;
        int input_size = config::input_size;
        
        // Create and return the NormalDataPredictor with loaded weights and biases
        auto predictor = NormalDataPredictor(weights, biases);
        std::cout << "NormalDataPredictor created with input size: " << input_size 
                  << " and hidden size: " << hidden_size << std::endl;
        return predictor;
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
        std::cout << "Observed value: " << observed_val << std::endl;
        observed_vals.push_back(observed_val);
        
        if (observed_vals.size() < predictor_config.lookback_len) {
            std::cout << "Not enough data yet. Returning false." << std::endl;
            return false;
        }

        // Prepare input for prediction
        std::vector<float> input(observed_vals.end() - predictor_config.lookback_len, observed_vals.end());
        std::cout << "Input for prediction: ";
        for (float val : input) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::vector<float> predicted = data_predictor.predict(input);
        if (predicted.empty()) {
            std::cerr << "Error: Empty prediction from data_predictor" << std::endl;
            return false;
        }
        float predicted_val = predicted[0];
        predicted_vals.push_back(predicted_val);
        std::cout << "Predicted value: " << predicted_val << std::endl;

        // Calculate error
        float error = std::abs(observed_val - predicted_val);
        predictive_errors.push_back(error);
        std::cout << "Prediction error: " << error << std::endl;

        // Generate threshold only if we have enough errors
        if (predictive_errors.size() >= predictor_config.lookback_len) {
            std::vector<float> past_errors(predictive_errors.end() - predictor_config.lookback_len, predictive_errors.end());
            std::cout << "Past errors for threshold generation: ";
            for (float err : past_errors) {
                std::cout << err << " ";
            }
            std::cout << std::endl;
            
            float threshold = generator.generate(past_errors, minimal_threshold);
            thresholds.push_back(threshold);
            std::cout << "Generated threshold: " << threshold << std::endl;

            // Determine if anomalous
            bool is_anomalous = (error > threshold);
            std::cout << "Is anomalous: " << (is_anomalous ? "true" : "false") << std::endl;

            // Update generator if necessary
            if (is_anomalous || threshold > minimal_threshold) {
                std::cout << "Updating generator..." << std::endl;
                generator.update(config::update_G_epoch, config::update_G_lr, past_errors);
            }

            // Log the result
            if (f_log.is_open()) {
                f_log << observed_val << "," << predicted_val << ","
                      << predicted_val - threshold << "," << predicted_val + threshold << ","
                      << is_anomalous << "," << error << "," << threshold << "\n";
            }

            return is_anomalous;
        }

        std::cout << "Not enough errors yet. Returning false." << std::endl;
        return false;

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