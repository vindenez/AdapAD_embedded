#include "adapad.hpp"
#include "matrix_utils.hpp"
#include "normal_data_prediction_error_calculator.hpp"
#include "config.hpp"
#include "optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

AdapAD::AdapAD(const PredictorConfig& predictor_config,
               const ValueRangeConfig& value_range_config,
               float minimal_threshold,
               const std::string& parameter_name)
    : value_range_config(value_range_config),
      predictor_config(predictor_config),
      minimal_threshold(minimal_threshold),
      parameter_name(parameter_name),
      config(Config::getInstance()),
      update_count(0) {
    
    // Initialize AdapAD components
    data_predictor.reset(new NormalDataPredictor(
        config.LSTM_size_layer,
        config.LSTM_size,
        predictor_config.lookback_len,
        predictor_config.prediction_len
    ));
    
    init_optimizer();
    
    generator.reset(new AnomalousThresholdGenerator(
        config.LSTM_size_layer,
        config.LSTM_size,
        predictor_config.lookback_len,
        predictor_config.prediction_len
    ));
    
    // Create parameter-specific log file name
    f_name = config.log_file_path + "/" + parameter_name + "_log.csv";
    
    // Ensure log directory exists
    mkdir(config.log_file_path.c_str(), 0777);
    
    // Initialize logging with the parameter-specific filename
    f_log.open(f_name);
    f_log << "observed,predicted,low,high,anomalous,err,threshold\n";
    f_log.close();

    // Create save directory if it doesn't exist
    mkdir(config.save_path.c_str(), 0777);  // UNIX-style directory creation
}

void AdapAD::init_optimizer() {
    std::unique_ptr<Optimizer> opt;
    
    if (config.optimizer_config.type == "sgd") {
        std::cout << "Creating SGD optimizer for " << parameter_name << " with:" << std::endl;
        std::cout << "  learning_rate: " << config.get_current_learning_rates().train << std::endl;
        std::cout << "  momentum: " << config.optimizer_config.sgd.momentum << std::endl;
        std::cout << "  weight_decay: " << config.optimizer_config.sgd.weight_decay << std::endl;
        
        opt.reset(new SGD(
            config.get_current_learning_rates().train,
            config.optimizer_config.sgd.momentum,
            config.optimizer_config.sgd.weight_decay
        ));
    } else {
        std::cout << "Creating Adam optimizer for " << parameter_name << " with:" << std::endl;
        std::cout << "  learning_rate: " << config.get_current_learning_rates().train << std::endl;
        std::cout << "  beta1: " << config.optimizer_config.adam.beta1 << std::endl;
        std::cout << "  beta2: " << config.optimizer_config.adam.beta2 << std::endl;
        
        opt.reset(new Adam(
            config.optimizer_config.adam.beta1,
            config.optimizer_config.adam.beta2,
            config.optimizer_config.adam.epsilon
        ));
    }
    
    data_predictor->set_optimizer(std::move(opt));
}

void AdapAD::set_training_data(const std::vector<float>& data) {
    observed_vals.clear();
    for (float val : data) {
        float normalized = normalize_data(val);
        observed_vals.push_back(normalized);
    }
}

std::vector<float> AdapAD::get_tail(const std::vector<float>& vec, size_t n) {
    if (vec.empty()) return std::vector<float>();
    
    size_t start = (n >= vec.size()) ? 0 : vec.size() - n;
    return std::vector<float>(vec.begin() + start, vec.end());
}

bool AdapAD::is_anomalous(float observed_val) {
    bool is_anomalous_ret = false;
    
    try {
        float normalized = normalize_data(observed_val);
        observed_vals.push_back(normalized);
        int supposed_anomalous_pos = observed_vals.size();

        // Check if we have enough data
        if (observed_vals.size() < predictor_config.lookback_len + 1) {
            std::cout << "Not enough data points yet. Have " << observed_vals.size() 
                      << ", need " << (predictor_config.lookback_len + 1) << std::endl;
            throw std::runtime_error("Not enough observed values");
        }

        std::vector<std::vector<std::vector<float>>> past_observations = prepare_data_for_prediction(supposed_anomalous_pos);
        float predicted_val = data_predictor->predict(past_observations);
        
        // Manage predicted_vals size
        if (predicted_vals.size() >= predictor_config.lookback_len * 2) {
            predicted_vals.erase(predicted_vals.begin());
        }
        predicted_vals.push_back(predicted_val);

        if (!is_inside_range(normalized)) {  // Changed from observed_val to normalized
            anomalies.push_back(supposed_anomalous_pos);
            is_anomalous_ret = true;
        }
        else {
            float threshold = minimal_threshold;
            
            // Only try to generate threshold if we have enough error history
            if (predictive_errors.size() >= predictor_config.lookback_len) {
                generator->eval();
                std::vector<float> past_predictive_errors = get_tail(predictive_errors, predictor_config.lookback_len);
                std::vector<std::vector<std::vector<float>>> input(1);
                input[0].resize(1);
                input[0][0] = past_predictive_errors;

                threshold = generator->generate(past_predictive_errors, minimal_threshold);
                threshold = std::max(threshold, minimal_threshold);
            }
            
            // Manage thresholds size
            if (!thresholds.empty() && thresholds.size() >= predictor_config.lookback_len * 2) {
                thresholds.erase(thresholds.begin());
            }
            thresholds.push_back(threshold);

            float prediction_error = NormalDataPredictionErrorCalculator::calc_error(predicted_val, normalized);  // Changed to normalized
            predictive_errors.push_back(prediction_error);
            
            if (prediction_error > threshold) {
                if (!is_default_normal()) {
                    is_anomalous_ret = true;
                    anomalies.push_back(supposed_anomalous_pos);
                }
            }
            
            data_predictor->update(config.epoch_update, config.lr_update, past_observations, {normalized});  // Changed to normalized

            if (is_anomalous_ret || threshold > minimal_threshold) {
                if (predictive_errors.size() >= predictor_config.lookback_len) {
                    std::vector<float> past_predictive_errors = get_tail(predictive_errors, predictor_config.lookback_len);
                    update_generator(past_predictive_errors, prediction_error);
                }
            }
        }
        
        logging(is_anomalous_ret);
        return is_anomalous_ret;

    } catch (const std::exception& e) {
        std::cerr << "Error in is_anomalous for " << parameter_name << ": " << e.what() << std::endl;
        throw;
    }
}

void AdapAD::update_generator(
    const std::vector<float>& past_errors, float recent_error) {
    
    generator->train();
    
    // Reshape past_errors to match PyTorch's reshape(1, -1)
    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].resize(1);
    reshaped_input[0][0] = past_errors;
    
    // Single forward pass before epoch loop
    auto output = generator->forward(reshaped_input);
    auto pred = generator->get_final_prediction(output);
    
    // Calculate initial loss
    float initial_loss = 0.0f;
    float diff = pred[0] - recent_error;
    initial_loss = diff * diff;
    
    // Training loop with early stopping based on initial loss
    std::vector<float> loss_history{initial_loss};
    for (int e = 0; e < config.get_current_epochs().update_generator; ++e) {
        if (e > 0 && initial_loss > loss_history.front()) {
            break;
        }
        
        generator->train_step(reshaped_input, {recent_error}, output, 
            config.get_current_learning_rates().update_generator);
    }
}

void AdapAD::clean() {
    size_t before_clean = get_current_memory();
    size_t window_size = predictor_config.lookback_len;
    
    // Only clean if we have more elements than the window size
    // AND all vectors have the same size
    if (predicted_vals.size() > window_size && 
        predicted_vals.size() == predictive_errors.size() &&
        predicted_vals.size() == thresholds.size()) {
            
        // Calculate how many elements to keep
        size_t keep_count = std::min(window_size, predicted_vals.size());
        size_t start_idx = predicted_vals.size() - keep_count;
        
        try {
            // Use clear and shrink_to_fit on temporary vectors
            {
                std::vector<float> new_predicted(predicted_vals.begin() + start_idx, predicted_vals.end());
                predicted_vals.swap(new_predicted);
                new_predicted.clear();
                new_predicted.shrink_to_fit();
            }
            
            if (!predictive_errors.empty()) {
                std::vector<float> new_errors(predictive_errors.begin() + start_idx, predictive_errors.end());
                predictive_errors.swap(new_errors);
                new_errors.clear();
                new_errors.shrink_to_fit();
            }
            
            if (!thresholds.empty()) {
                std::vector<float> new_thresholds(thresholds.begin() + start_idx, thresholds.end());
                thresholds.swap(new_thresholds);
                new_thresholds.clear();
                new_thresholds.shrink_to_fit();
            }
            
            // Force capacity to match size
            predicted_vals.shrink_to_fit();
            predictive_errors.shrink_to_fit();
            thresholds.shrink_to_fit();
            
        } catch (const std::exception& e) {
            std::cerr << "Error in clean(): " << e.what() << std::endl;
        }
    }
    
    // Clean LSTM caches
    data_predictor->clear_temporary_cache();
    generator->clear_temporary_cache();
    
    size_t after_clean = get_current_memory();
    if (after_clean > before_clean) {
        std::cout << "Memory for " << parameter_name 
                  << " - Before: " << before_clean / 1024.0 
                  << " MB, After: " << after_clean / 1024.0 
                  << " MB" << std::endl;
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
    
    // Replace anomalous values with predictions
    for (int i = 0; i < predictor_config.lookback_len; i++) {
        int check_pos = predictor_config.lookback_len - i - 1;
        if (!is_inside_range(x_temp[check_pos])) {
            x_temp[check_pos] = predicted_vals[predicted_vals.size() - check_pos - 1];
        }
    }
    
    // Reshape for LSTM input
    std::vector<std::vector<std::vector<float>>> input(1);
    input[0].resize(1);
    input[0][0] = x_temp;
    
    return input;
}

void AdapAD::train() {
    // Add debug print
    std::cout << "Training with optimizer: " << config.optimizer_config.type << std::endl;
    std::cout << "Using epochs: " << config.get_current_epochs().train << std::endl;
    std::cout << "Using learning rate: " << config.get_current_learning_rates().train << std::endl;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Train data predictor and get training data
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>> 
        training_data = data_predictor->train(
            config.get_current_epochs().train,
            config.get_current_learning_rates().train,
            observed_vals);
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
    generator->train(config.epoch_train, config.lr_train, predictive_errors);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Save after initial training if enabled
    if (config.save_enabled) {
        try {
            save_models();
            update_count = 0;  // Reset counter after saving
        } catch (const std::exception& e) {
            std::cerr << "Failed to save initial model state: " << e.what() << std::endl;
        }
    }

    // After training is complete, reset states like we do in load_models
    std::cout << "Resetting model states after training..." << std::endl;
    reset_model_states();
    
    // Clear and reinitialize caches
    data_predictor->initialize_layer_cache();
    generator->initialize_layer_cache();
    
    // Clear history vectors
    predicted_vals.clear();
    predictive_errors.clear();
    thresholds.clear();
    
    // Keep only the most recent lookback_len values in observed_vals
    if (observed_vals.size() > predictor_config.lookback_len) {
        std::vector<float> recent_vals(
            observed_vals.end() - predictor_config.lookback_len,
            observed_vals.end()
        );
        observed_vals = std::move(recent_vals);  // Use move for efficiency
    }
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
    for (int epoch = 0; epoch < config.epoch_train; epoch++) {
        for (size_t i = 0; i < batch_x.size(); i++) {
            auto input = std::vector<std::vector<std::vector<float>>>(1);
            input[0].push_back(batch_x[i]);
            auto target = std::vector<float>{batch_y[i]};
            
            // Add forward pass and pass output to train_step
            auto output = generator->forward(input);
            generator->train_step(input, target, output, config.lr_train);
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

void AdapAD::save_models() {
    try {
        // Create directory if it doesn't exist
        if (mkdir(config.save_path.c_str(), 0777) == -1) {
            if (errno != EEXIST) {
                throw std::runtime_error("Failed to create save directory");
            }
        }
        
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        
        // Remove previous model file for this parameter if it exists
        DIR* dir = opendir(config.save_path.c_str());
        if (dir != nullptr) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                // Check if file is a previous save for this parameter
                if (filename.find(parameter_name + "_model_") == 0 && 
                    filename.find(".bin") != std::string::npos) {
                    std::string old_file = config.save_path + "/" + filename;
                    if (remove(old_file.c_str()) != 0) {
                        std::cerr << "Warning: Could not remove old model file: " << old_file << std::endl;
                    } else {
                        std::cout << "Updated model save file for parameter: " << parameter_name << std::endl;
                    }
                }
            }
            closedir(dir);
        }
        
        // Create new file path with parameter name and timestamp
        std::string save_file = config.save_path + "/" + 
                               parameter_name + 
                               "_model_" + 
                               timestamp.str() + 
                               ".bin";
        
        std::ofstream file(save_file, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + save_file);
        }

        // Save metadata
        file.write(reinterpret_cast<const char*>(&minimal_threshold), sizeof(float));
        file.write(reinterpret_cast<const char*>(&value_range_config.lower_bound), sizeof(float));
        file.write(reinterpret_cast<const char*>(&value_range_config.upper_bound), sizeof(float));

        // Save layer cache states
        data_predictor->save_layer_cache(file);
        generator->save_layer_cache(file);

        // Save predictor weights and biases directly to file
        data_predictor->save_weights(file);
        data_predictor->save_biases(file);
        
        // Save generator weights and biases directly to file
        generator->save_weights(file);
        generator->save_biases(file);
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving model state: " << e.what() << std::endl;
        throw;
    }
}

void AdapAD::reset_model_states() {
    if (data_predictor) {
        data_predictor->reset_states();
    }
    if (generator) {
        generator->reset_states();
    }
}

void AdapAD::load_models(const std::string& timestamp, const std::vector<float>& initial_data) {
    try {
        if (initial_data.size() < predictor_config.lookback_len) {
            throw std::runtime_error("Not enough initial data points provided. Need at least " + 
                                   std::to_string(predictor_config.lookback_len) + " points.");
        }

        std::string load_file = config.save_path + "/" + parameter_name + "_model_" + timestamp + ".bin";
        
        if (access(load_file.c_str(), F_OK) == -1) {
            throw std::runtime_error("Model file does not exist: " + load_file);
        }

        std::ifstream file(load_file, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + load_file);
        }

        try {
            std::cout << "Loading metadata..." << std::endl;
            // Load metadata first
            float temp_minimal_threshold;
            float temp_lower_bound;
            float temp_upper_bound;
            
            file.read(reinterpret_cast<char*>(&temp_minimal_threshold), sizeof(float));
            file.read(reinterpret_cast<char*>(&temp_lower_bound), sizeof(float));
            file.read(reinterpret_cast<char*>(&temp_upper_bound), sizeof(float));

            if (!file.good()) {
                throw std::runtime_error("Failed to read metadata");
            }

            std::cout << "Updating configuration values..." << std::endl;
            // Only update values after successful read
            minimal_threshold = temp_minimal_threshold;
            value_range_config.lower_bound = temp_lower_bound;
            value_range_config.upper_bound = temp_upper_bound;

            std::cout << "Initializing layer caches..." << std::endl;
            // Initialize layer caches
            data_predictor->initialize_layer_cache();
            generator->initialize_layer_cache();

            std::cout << "Loading layer cache states..." << std::endl;
            // Load layer cache states
            try {
                data_predictor->load_layer_cache(file);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to load data predictor cache: " + std::string(e.what()));
            }

            try {
                generator->load_layer_cache(file);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to load generator cache: " + std::string(e.what()));
            }

            std::cout << "Loading weights and biases..." << std::endl;
            // Load weights and biases
            try {
                data_predictor->load_weights(file);
                data_predictor->load_biases(file);
                generator->load_weights(file);
                generator->load_biases(file);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to load weights/biases: " + std::string(e.what()));
            }

            std::cout << "Resetting model states..." << std::endl;
            // Reset states after loading
            reset_model_states();
            
            std::cout << "Initializing observed values..." << std::endl;
            // Initialize observed_vals with exactly lookback_len points
            observed_vals.clear();
            for (size_t i = 0; i < predictor_config.lookback_len; i++) {
                float normalized = normalize_data(initial_data[i]);
                observed_vals.push_back(normalized);
            }
            
            // Initialize other vectors
            predicted_vals.clear();
            predictive_errors.clear();
            thresholds.clear();
            
            std::cout << "Successfully loaded model state for " << parameter_name << std::endl;
            
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("Error during model loading: " + std::string(e.what()));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading model state: " << e.what() << std::endl;
        throw;
    }
}

std::string AdapAD::get_state_filename() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << config.save_path << "model_state_" 
       << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") 
       << ".bin";
    return ss.str();
}

void AdapAD::clean_old_saves(size_t keep_count) {
    try {
        std::vector<std::string> files;
        DIR* dir = opendir(config.save_path.c_str());
        if (dir != nullptr) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                if (filename.size() > 4 && 
                    filename.substr(filename.size() - 4) == ".bin") {
                    files.push_back(config.save_path + filename);
                }
            }
            closedir(dir);
            
            // Sort by filename (which includes timestamp)
            std::sort(files.begin(), files.end());
            
            // Remove older files, keeping only the most recent ones
            while (files.size() > keep_count) {
                if (remove(files.front().c_str()) != 0) {
                    std::cerr << "Error deleting file: " << files.front() << std::endl;
                }
                files.erase(files.begin());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to clean old saves: " << e.what() << std::endl;
    }
}

bool AdapAD::has_saved_model() const {
    DIR* dir = opendir(config.save_path.c_str());
    if (dir == nullptr) {
        return false;
    }

    bool found = false;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        // Check if file is a save for this parameter
        if (filename.find(parameter_name + "_model_") == 0 && 
            filename.find(".bin") != std::string::npos) {
            found = true;
            break;
        }
    }
    closedir(dir);
    return found;
}

void AdapAD::load_latest_model(const std::vector<float>& initial_data) {
    DIR* dir = opendir(config.save_path.c_str());
    if (dir == nullptr) {
        throw std::runtime_error("Could not open save directory");
    }

    std::string latest_file;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.find(parameter_name + "_model_") == 0 && 
            filename.find(".bin") != std::string::npos) {
            latest_file = filename;
        }
    }
    closedir(dir);

    if (!latest_file.empty()) {
        std::string timestamp = latest_file.substr(parameter_name.length() + 7, 15);
        load_models(timestamp, initial_data);
    } else {
        throw std::runtime_error("No saved model found for " + parameter_name);
    }
}