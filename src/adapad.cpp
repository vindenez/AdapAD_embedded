#include "adapad.hpp"
#include "matrix_utils.hpp"
#include "normal_data_prediction_error_calculator.hpp"
#include "config.hpp"
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

void AdapAD::set_training_data(const std::vector<float>& data) {
    observed_vals.clear();
    for (float val : data) {
        float normalized = normalize_data(val);
        observed_vals.push_back(normalized);
    }
}

bool AdapAD::is_anomalous(float observed_val) {
    bool is_anomalous_ret = false;
    
    try {
        float normalized = normalize_data(observed_val);
        
        observed_vals.push_back(normalized);
        
        // We need lookback_len + 1 points: lookback_len for the window and 1 for prediction
        if (observed_vals.size() < predictor_config.lookback_len + 1) {
            std::cout << "Not enough data points yet. Have " << observed_vals.size() 
                      << ", need " << (predictor_config.lookback_len + 1) << std::endl;
            throw std::runtime_error("Not enough observed values");
        }
        
        auto input_data = prepare_data_for_prediction(observed_vals.size() - 1);
        
        reset_model_states();  // Reset states before each prediction
        auto prediction = data_predictor->predict(input_data);
        
        float threshold;
        if (predictive_errors.size() >= predictor_config.lookback_len) {
            // Create vector of past errors for threshold generation
            std::vector<float> past_errors(
                predictive_errors.end() - predictor_config.lookback_len,
                predictive_errors.end()
            );
            threshold = generator->generate(past_errors, minimal_threshold);
        } else {
            // Use minimal threshold for initial predictions
            threshold = minimal_threshold;
        }
        
        // Validate vector sizes before push_back
        if (predicted_vals.size() >= predictor_config.lookback_len * 2) {
            predicted_vals.erase(predicted_vals.begin());
        }
        predicted_vals.push_back(prediction);
        
        // Calculate error in normalized space to match thresholds
        float prediction_error = NormalDataPredictionErrorCalculator::calc_error(
            prediction, normalized);  
        
        predictive_errors.push_back(prediction_error);
        
        // Check range first
        if (!is_inside_range(normalized)) {
            is_anomalous_ret = true;
            anomalies.push_back(observed_vals.size());
        } else {
            // Only process thresholds and errors for in-range values
            if (prediction_error > threshold && !is_default_normal()) {
                is_anomalous_ret = true;
                anomalies.push_back(observed_vals.size());
            }
            
            // Update models only for in-range values
            data_predictor->update(config.epoch_update, config.lr_update,
                                input_data, {normalized});
            
            if (is_anomalous_ret || threshold > minimal_threshold) {
                // Only update generator if we have enough errors
                if (predictive_errors.size() >= predictor_config.lookback_len) {
                    std::vector<float> past_errors(
                        predictive_errors.end() - predictor_config.lookback_len,
                        predictive_errors.end()
                    );
                    update_generator(past_errors, prediction_error);
                }
            }
        }
        
        // Log results
        f_log.open(f_name, std::ios_base::app);
        f_log << observed_val << ","
              << reverse_normalized_data(prediction) << ","
              << reverse_normalized_data(prediction - threshold) << ","
              << reverse_normalized_data(prediction + threshold) << ","
              << (is_anomalous_ret ? "True" : "False") << ","
              << (predictive_errors.empty() ? 0.0f : predictive_errors.back()) << ","
              << threshold << "\n";
        f_log.close();
        
        // Periodic state saving (only if enabled)
        if (config.save_enabled) {
            update_count++;
            if (update_count >= config.save_interval) {
                try {
                    save_models();
                    update_count = 0;  // Reset counter after saving
                } catch (const std::exception& e) {
                    std::cerr << "Failed to save model state: " << e.what() << std::endl;
                }
            }
        }
                
        return is_anomalous_ret;
    } catch (const std::exception& e) {
        std::cerr << "Error in is_anomalous for " << parameter_name << ": " << e.what() << std::endl;
        throw;
    }
}

void AdapAD::update_generator(
    const std::vector<float>& past_errors, float recent_error) {
    
    std::vector<float> loss_history;
    generator->train();
    
    // Single update with early stopping
    for (int e = 0; e < config.update_G_epoch; ++e) {
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
        
        generator->train_step(reshaped_input, {recent_error}, config.update_G_lr);
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
    
    // Only try to use predicted values if we have them
    if (!predicted_vals.empty() && predicted_vals.size() >= predictor_config.lookback_len) {
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
    }
    
    // Create tensor matching PyTorch's reshape(1, -1) 
    std::vector<std::vector<std::vector<float>>> input_tensor(1);
    input_tensor[0].resize(1);
    input_tensor[0][0] = x_temp;
    
    return input_tensor;
}

void AdapAD::train() {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Train data predictor and get training data
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>> 
        training_data = data_predictor->train(config.epoch_train, config.lr_train, observed_vals);
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
            
            generator->train_step(input, target, config.lr_train);
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