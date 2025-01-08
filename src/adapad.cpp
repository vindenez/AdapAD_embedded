#include "adapad.hpp"
#include "matrix_utils.hpp"
#include "normal_data_prediction_error_calculator.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

AdapAD::AdapAD(const PredictorConfig& predictor_config,
               const ValueRangeConfig& value_range_config,
               float minimal_threshold)
    : value_range_config(value_range_config),
      predictor_config(predictor_config),
      minimal_threshold(minimal_threshold) {
    
    // Initialize learning components
    data_predictor.reset(new NormalDataPredictor(
        config::LSTM_size_layer,
        config::LSTM_size,
        predictor_config.lookback_len,
        predictor_config.prediction_len
    ));
    
    generator.reset(new AnomalousThresholdGenerator(
        config::LSTM_size_layer,
        config::LSTM_size,
        predictor_config.lookback_len,
        predictor_config.prediction_len
    ));
    
    // Initialize logging with the specified filename
    f_name = config::log_file_path;  // Use the path from config
    f_log.open(f_name);
    f_log << "observed,predicted,low,high,anomalous,err,threshold\n";
    f_log.close();
    
}

void AdapAD::set_training_data(const std::vector<float>& data) {
    observed_vals.clear();
    for (float val : data) {
        float normalized = normalize_data(val);
        observed_vals.push_back(normalized);
    }
}

bool AdapAD::is_anomalous(float observed_val) {
    bool is_anomalous_ret = false;
    float normalized = normalize_data(observed_val);
    
    observed_vals.push_back(normalized);
    
    try {
        // Validate vector sizes before operations
        if (observed_vals.size() < predictor_config.lookback_len + 1) {
            throw std::runtime_error("Not enough observed values");
        }

        // Validate past_observations dimensions
        auto past_observations = prepare_data_for_prediction(observed_vals.size());
        if (past_observations.empty() || past_observations[0].empty() || 
            past_observations[0][0].size() != predictor_config.lookback_len) {
            throw std::runtime_error("Invalid past_observations dimensions");
        }

        // Make prediction
        data_predictor->eval();
        auto predicted_val = data_predictor->predict(past_observations);
        
        // Validate vector sizes before push_back
        if (predicted_vals.size() >= predictor_config.lookback_len * 2) {
            predicted_vals.erase(predicted_vals.begin());
        }
        predicted_vals.push_back(predicted_val);
        
        // Calculate error in normalized space to match thresholds
        float prediction_error = NormalDataPredictionErrorCalculator::calc_error(
            predicted_val, normalized);  
        
        predictive_errors.push_back(prediction_error);
        
        // Check range first
        if (!is_inside_range(normalized)) {
            is_anomalous_ret = true;
            anomalies.push_back(observed_vals.size());
        } else {
            // Only process thresholds and errors for in-range values
            float threshold = minimal_threshold;
            
            if (static_cast<int>(predictive_errors.size()) >= predictor_config.lookback_len) {
                auto past_errors = std::vector<float>(
                    predictive_errors.end() - predictor_config.lookback_len,
                    predictive_errors.end());
                
                threshold = generator->generate(past_errors, minimal_threshold);
                
                if (prediction_error > threshold && !is_default_normal()) {
                    is_anomalous_ret = true;
                    anomalies.push_back(observed_vals.size());
                }
                
                // Update models only for in-range values
                data_predictor->update(config::epoch_update, config::lr_update,
                                    past_observations, {normalized});
                
                if (is_anomalous_ret || threshold > minimal_threshold) {
                    update_generator(past_errors, prediction_error);
                }
            }
            thresholds.push_back(threshold);
        }
        
        // Log results
        f_log.open(f_name, std::ios_base::app);
        f_log << observed_val << ","
              << reverse_normalized_data(predicted_val) << ","
              << reverse_normalized_data(predicted_val - (thresholds.empty() ? minimal_threshold : thresholds.back())) << ","
              << reverse_normalized_data(predicted_val + (thresholds.empty() ? minimal_threshold : thresholds.back())) << ","
              << (is_anomalous_ret ? "True" : "False") << ","
              << (predictive_errors.empty() ? 0.0f : predictive_errors.back()) << ","
              << (thresholds.empty() ? minimal_threshold : thresholds.back()) << "\n";
        f_log.close();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in is_anomalous: " << e.what() << std::endl;
        throw;
    }
    
    return is_anomalous_ret;
}

void AdapAD::update_generator(
    const std::vector<float>& past_errors, float recent_error) {
    
    std::vector<float> loss_history;
    generator->train();
    
    // Single update with early stopping
    for (int e = 0; e < config::update_G_epoch; ++e) {
        // Reshape past_errors to match PyTorch's reshape(1, -1)
        std::vector<std::vector<std::vector<float>>> reshaped_input(1);
        reshaped_input[0].resize(1);
        reshaped_input[0][0] = past_errors;
        
        auto output = generator->forward(reshaped_input);
        auto pred = generator->get_final_prediction(output);
        
        // Calculate MSE loss
        float current_loss = 0.0f;
        float diff = pred[0] - recent_error;
        current_loss = diff * diff;
        
        // Early stopping check
        if (!loss_history.empty() && current_loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(current_loss);
        
        generator->train_step(reshaped_input, {recent_error}, config::update_G_lr);
    }
}

void AdapAD::clean() {
    size_t window_size = predictor_config.lookback_len;
    
    // Only clean if we have more elements than the window size
    // AND all vectors have the same size
    if (predicted_vals.size() > window_size && 
        predicted_vals.size() == predictive_errors.size() &&
        predicted_vals.size() == thresholds.size()) {
            
        // Calculate how many elements to keep
        size_t keep_count = std::min(window_size, predicted_vals.size());
        
        // Calculate starting point for keeping elements
        size_t start_idx = predicted_vals.size() - keep_count;
        
        try {
            // Create temporary vectors with the elements we want to keep
            std::vector<float> new_predicted(predicted_vals.begin() + start_idx, predicted_vals.end());
            
            // Only clean other vectors if they have elements
            if (!predictive_errors.empty()) {
                std::vector<float> new_errors(predictive_errors.begin() + start_idx, predictive_errors.end());
                predictive_errors.swap(new_errors);
            }
            
            if (!thresholds.empty()) {
                std::vector<float> new_thresholds(thresholds.begin() + start_idx, thresholds.end());
                thresholds.swap(new_thresholds);
            }
            
            predicted_vals.swap(new_predicted);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in clean(): " << e.what() << std::endl;
            // Continue without cleaning if there's an error
        }
    }
}

float AdapAD::normalize_data(float val) {
    return (val - value_range_config.lower_bound) / 
           (value_range_config.upper_bound - value_range_config.lower_bound);
}

float AdapAD::reverse_normalized_data(float val) {
    return val * (value_range_config.upper_bound - value_range_config.lower_bound) + 
           value_range_config.lower_bound;
}

bool AdapAD::is_inside_range(float val) {
    float denormalized = reverse_normalized_data(val);
    return denormalized >= value_range_config.lower_bound && 
           denormalized <= value_range_config.upper_bound;
}

bool AdapAD::is_default_normal() {
    size_t window_size = std::min(predictor_config.train_size, 
                                 (int)observed_vals.size());
    auto recent_vals = std::vector<float>(
        observed_vals.end() - window_size,
        observed_vals.end()
    );
    
    int cnt = 0;
    for (float val : recent_vals) {
        if (!is_inside_range(val)) {
            cnt++;
        }
    }
    
    return cnt > predictor_config.train_size / 2;
}

void AdapAD::logging(bool is_anomalous_ret) {
    f_log.open(f_name, std::ios_base::app);
    
    float current_threshold = thresholds.back();
    float current_predicted = predicted_vals.back();
    float current_observed = observed_vals.back();
    float current_error = predictive_errors.back();
    
    f_log << reverse_normalized_data(current_observed) << ","
          << reverse_normalized_data(current_predicted) << ","
          << reverse_normalized_data(current_predicted - current_threshold) << ","
          << reverse_normalized_data(current_predicted + current_threshold) << ","
          << (is_anomalous_ret ? "True" : "False") << ","
          << current_error << ","
          << current_threshold << "\n";
    
    f_log.close();
}

std::vector<std::vector<std::vector<float>>> 
AdapAD::prepare_data_for_prediction(size_t supposed_anomalous_pos) {
    // Get lookback window 
    std::vector<float> x_temp(
        observed_vals.end() - predictor_config.lookback_len - 1,
        observed_vals.end() - 1
    );
    
    // Get predicted values 
    std::vector<float> predicted(
        predicted_vals.end() - predictor_config.lookback_len,
        predicted_vals.end()
    );
    
    // Replace out-of-range values 
    for (int i = 0; i < predictor_config.lookback_len; ++i) {
        if (!is_inside_range(x_temp[x_temp.size() - i - 1])) {
            x_temp[x_temp.size() - i - 1] = predicted[predicted.size() - i - 1];
        }
    }
    
    // Create tensor matching PyTorch's reshape(1, -1) 
    std::vector<std::vector<std::vector<float>>> input_tensor(1);
    input_tensor[0].resize(1);
    input_tensor[0][0] = x_temp;
    
    return input_tensor;
}

void AdapAD::train() {
    std::cout << "Starting predictor training with epochs=" << config::epoch_train 
              << ", lr=" << config::lr_train << std::endl;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Train data predictor and get training data
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>> 
        training_data = data_predictor->train(config::epoch_train, config::lr_train, observed_vals);
    auto& trainX = training_data.first;
    auto& trainY = training_data.second;
    
    // Calculate and store predicted values for training data
    predicted_vals.clear();
    for (const auto& x : trainX) {
        std::vector<std::vector<std::vector<float>>> input_tensor(1);
        input_tensor[0].resize(1);
        input_tensor[0][0] = x[0];
        
        auto pred = data_predictor->predict(input_tensor);
        predicted_vals.push_back(pred);
        
        // Log training predictions without thresholds
        f_log.open(f_name, std::ios_base::app);
        f_log << reverse_normalized_data(observed_vals[predicted_vals.size()-1]) << ","
              << reverse_normalized_data(pred) << ",,,,," << "\n";
        f_log.close();
    }
    
    // Calculate prediction errors for training data
    predictive_errors.clear();
    for (size_t i = 0; i < trainY.size(); i++) {
        float error = std::abs(trainY[i] - predicted_vals[i]);
        predictive_errors.push_back(error);
    }
    
    // Train generator
    generator->reset_states();
    generator->train(config::epoch_train, config::lr_train, predictive_errors);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Training complete in " << elapsed.count() << " seconds" << std::endl;
}

void AdapAD::learn_error_pattern(
    const std::vector<std::vector<std::vector<float>>>& trainX,
    const std::vector<float>& trainY) {
    
    // Calculate predictions
    predicted_vals.clear();
    for (size_t i = 0; i < trainX.size(); i++) {
        auto reshaped_input = std::vector<std::vector<std::vector<float>>>(1);
        reshaped_input[0] = trainX[i];
        float pred = data_predictor->predict(reshaped_input);
        predicted_vals.push_back(pred);
    }

    // Get tail of predicted values
    auto recent_predicted = std::vector<float>(
        predicted_vals.end() - trainY.size(),
        predicted_vals.end()
    );
    
    // Calculate errors
    predictive_errors = NormalDataPredictionErrorCalculator::calc_error(
        trainY, recent_predicted);

    // Train generator using batch learning approach
    std::pair<std::vector<std::vector<float>>, std::vector<float>> 
        batch_data = create_sliding_windows(
            predictive_errors, 
            predictor_config.lookback_len,
            predictor_config.prediction_len
        );
    auto& batch_x = batch_data.first;
    auto& batch_y = batch_data.second;
    
    generator->reset_states();
    for (int epoch = 0; epoch < config::epoch_train; epoch++) {
        for (size_t i = 0; i < batch_x.size(); i++) {
            auto input = std::vector<std::vector<std::vector<float>>>(1);
            input[0].push_back(batch_x[i]);
            auto target = std::vector<float>{batch_y[i]};
            
            generator->train_step(input, target, config::lr_train);
        }
    }

    // Log results
    for (size_t i = 0; i < trainY.size(); i++) {
        f_log.open(f_name, std::ios_base::app);
        f_log << reverse_normalized_data(trainY[i]) << ","
              << reverse_normalized_data(predicted_vals[i]) << ",,,,\n";
        f_log.close();
    }
}

float AdapAD::simplify_error(const std::vector<float>& errors, float N_sigma) {
    if (errors.empty()) {
        return 0.0f;
    }

    // Calculate mean
    float sum = std::accumulate(errors.begin(), errors.end(), 0.0f);
    float mean = sum / errors.size();

    if (N_sigma == 0) {
        return mean;
    }

    // Calculate standard deviation
    float sq_sum = std::inner_product(errors.begin(), errors.end(), 
                                    errors.begin(), 0.0f);
    float std_dev = std::sqrt(sq_sum / errors.size() - mean * mean);

    return mean + N_sigma * std_dev;
}