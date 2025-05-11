#include "adapad.hpp"
#include "config.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

AdapAD::AdapAD(const PredictorConfig &predictor_config, const ValueRangeConfig &value_range_config,
               float minimal_threshold, const std::string &parameter_name)
    : value_range_config(value_range_config), predictor_config(predictor_config),
      minimal_threshold(minimal_threshold), config(Config::getInstance()),
      parameter_name(parameter_name), update_count(0) {

    // Initialize AdapAD components
    data_predictor.reset(new NormalDataPredictor(config.LSTM_size_layer, config.LSTM_size,
                                                 predictor_config.lookback_len,
                                                 predictor_config.prediction_len));

    generator.reset(new AnomalousThresholdGenerator(config.LSTM_size_layer, config.LSTM_size,
                                                    predictor_config.lookback_len,
                                                    predictor_config.prediction_len));

    // Create parameter-specific log file name
    f_name = config.log_file_path + "/" + parameter_name + "_log.csv";

    // Ensure log directory exists
    mkdir(config.log_file_path.c_str(), 0777);

    // Initialize logging with the parameter-specific filename
    f_log.open(f_name);
    f_log << "observed,predicted,low,high,anomalous,err,threshold\n";
    f_log.close();

    // Create save directory if it doesn't exist
    mkdir(config.save_path.c_str(), 0777);
}

void AdapAD::set_training_data(const std::vector<float> &data) {
    observed_vals.clear();
    for (float val : data) {
        float normalized = normalize_data(val);
        observed_vals.push_back(normalized);
    }
}
bool AdapAD::is_anomalous(float observed_val) {
    bool is_anomalous_ret = false;
    float normalized = normalize_data(observed_val);

    // First add the new value
    observed_vals.push_back(normalized);

    // Then remove the oldest entry if we have more than lookback_len values
    if (observed_vals.size() >
        predictor_config.lookback_len + 1) { // +1 to ensure we have enough for prediction
        observed_vals.erase(observed_vals.begin());
    }

    try {
        // Validate vector sizes before operations
        if (observed_vals.size() < predictor_config.lookback_len + 1) {
            throw std::runtime_error("Not enough observed values");
        }

        float normalized_val = normalize_data(observed_val);

        // Get the previous lookback_len values for prediction
        auto input_data = prepare_data_for_prediction(observed_vals.size() - 1);

        // Use the new forward method that returns the prediction directly
        std::vector<float> prediction = data_predictor->forward(input_data);
        
        // Get the predicted value (first/only element in the prediction vector)
        float predicted_val = reverse_normalized_data(prediction[0]);
        predicted_vals.push_back(predicted_val);

        // Calculate prediction error
        float prediction_error = calc_error(predicted_val, observed_val);
        predictive_errors.push_back(prediction_error);

        // Check range and handle out-of-range values
        if (!is_inside_range(normalized)) {
            is_anomalous_ret = true;
            anomalies.push_back(observed_vals.size());

            // Log out-of-range value as -999 (matching Python implementation)
            f_log.open(f_name, std::ios_base::app);
            f_log << "-999," << predicted_val << ","
                  << predicted_val - minimal_threshold << ","
                  << predicted_val + minimal_threshold << ","
                  << "True" << "," << prediction_error << "," << minimal_threshold << "\n";
            f_log.close();
        } else {
            // Only process thresholds and errors for in-range values
            float threshold = minimal_threshold;

            if (static_cast<int>(predictive_errors.size()) >= predictor_config.lookback_len) {
                // Get the last lookback_len prediction errors
                auto past_errors =
                    std::vector<float>(predictive_errors.end() - predictor_config.lookback_len,
                                       predictive_errors.end());

                // Create generator input for threshold generation
                std::vector<std::vector<std::vector<float>>> generator_input(1);
                generator_input[0].resize(1);
                generator_input[0][0] = past_errors;
                
                // Generate threshold using a single forward call
                threshold = generator->generate(past_errors, minimal_threshold);

                if (prediction_error > threshold && !is_default_normal()) {
                    is_anomalous_ret = true;
                    anomalies.push_back(observed_vals.size());
                }

                // Only update models if not in default normal state
                if (!is_default_normal()) {
                    // Update data predictor
                    data_predictor->update(predictor_config.epoch_update,
                                          predictor_config.lr_update, 
                                          input_data,
                                          {normalized_val});

                    // Update generator if anomalous or threshold is significant
                    if (is_anomalous_ret || threshold > minimal_threshold) {
                        generator->update(predictor_config.epoch_update_generator,
                                         predictor_config.lr_update_generator, 
                                         past_errors,
                                         prediction_error);
                    }
                }
            }

            thresholds.push_back(threshold);

            // Log in-range value
            f_log.open(f_name, std::ios_base::app);
            f_log << observed_val << "," << predicted_val << ","
                  << predicted_val - (thresholds.empty() ? minimal_threshold : thresholds.back()) << ","
                  << predicted_val + (thresholds.empty() ? minimal_threshold : thresholds.back()) << ","
                  << (is_anomalous_ret ? "True" : "False") << ","
                  << (predictive_errors.empty() ? 0.0f : predictive_errors.back()) << ","
                  << (thresholds.empty() ? minimal_threshold : thresholds.back()) << "\n";
            f_log.close();
        }

        // Check if we should save the model based on update count
        if (config.save_enabled && ++update_count >= config.save_interval) {
            try {
                save_model(); // Changed from save_model() to match class declaration
                update_count = 0; 
            } catch (const std::exception &e) {
                std::cerr << "Failed to save model state: " << e.what() << std::endl;
            }
        }

        clean();

    } catch (const std::exception &e) {
        std::cerr << "Error in is_anomalous: " << e.what() << std::endl;
        throw;
    }

    return is_anomalous_ret;
}

void AdapAD::clean() {
    size_t window_size = predictor_config.lookback_len;

    if (observed_vals.size() > window_size) {
        std::vector<float> recent_vals(observed_vals.end() - window_size, observed_vals.end());
        observed_vals.swap(recent_vals);
        observed_vals.shrink_to_fit();
    }

    if (predicted_vals.size() > window_size) {
        std::vector<float> recent_preds(predicted_vals.end() - window_size, predicted_vals.end());
        predicted_vals.swap(recent_preds);
        predicted_vals.shrink_to_fit();
    }

    if (predictive_errors.size() > window_size) {
        std::vector<float> recent_errors(predictive_errors.end() - window_size,
                                         predictive_errors.end());
        predictive_errors.swap(recent_errors);
        predictive_errors.shrink_to_fit();
    }

    if (thresholds.size() > window_size) {
        std::vector<float> recent_thresholds(thresholds.end() - window_size, thresholds.end());
        thresholds.swap(recent_thresholds);
        thresholds.shrink_to_fit();
    }

    if (anomalies.size() > 100) {
        std::vector<size_t> recent_anomalies(anomalies.end() - 100, anomalies.end());
        anomalies.swap(recent_anomalies);
        anomalies.shrink_to_fit();
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
    size_t window_size = std::min(predictor_config.train_size, (int)observed_vals.size());
    auto recent_vals = std::vector<float>(observed_vals.end() - window_size, observed_vals.end());

    int cnt = 0;
    for (float val : recent_vals) {
        if (!is_inside_range(val)) {
            cnt++;
        }
    }

    return cnt > predictor_config.train_size / 2;
}

float AdapAD::calc_error(float predicted_val, float observed_val) {
    float diff = predicted_val - observed_val;
    return diff * diff;
}

std::vector<float> AdapAD::calc_error(const std::vector<float> &observed_vals,
                                        const std::vector<float> &predicted_vals) {

    if (observed_vals.size() != predicted_vals.size()) {
        throw std::runtime_error("Observed and predicted values must have same length");
    }

    std::vector<float> errors;
    errors.reserve(observed_vals.size());

    for (std::size_t i = 0; i < observed_vals.size(); i++) {
        errors.push_back(calc_error(predicted_vals[i], observed_vals[i]));
    }

    return errors;
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
          << (is_anomalous_ret ? "True" : "False") << "," << current_error << ","
          << current_threshold << "\n";

    f_log.close();
}

std::vector<std::vector<std::vector<float>>>
AdapAD::prepare_data_for_prediction(size_t supposed_anomalous_pos) {
    // Get lookback window
    std::vector<float> x_temp(observed_vals.end() - predictor_config.lookback_len - 1,
                              observed_vals.end() - 1);

    // Create tensor matching PyTorch's reshape(1, -1)
    std::vector<std::vector<std::vector<float>>> input_tensor(1);
    input_tensor[0].resize(1);
    input_tensor[0][0] = x_temp;

    return input_tensor;
}

void AdapAD::train() {
    // Reset states before training
    data_predictor->reset_states();
    generator->reset_states();

    // Train data predictor and get training data
    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>> training_data = data_predictor->train(config.epoch_train, config.lr_train, observed_vals);
    auto &trainX = training_data.first;
    auto &trainY = training_data.second;

    // Calculate and store predicted values for training data
    predicted_vals.clear();
    for (const auto &x : trainX) {
        std::vector<std::vector<std::vector<float>>> input_tensor(1);
        input_tensor[0].resize(1);
        input_tensor[0][0] = x[0];

        std::vector<float> prediction = data_predictor->forward(input_tensor);
        float pred_value = prediction[0];
        predicted_vals.push_back(pred_value);

        // Log training predictions without thresholds
        f_log.open(f_name, std::ios_base::app);
        f_log << reverse_normalized_data(observed_vals[predicted_vals.size() - 1]) << ","
              << reverse_normalized_data(pred_value) << ",,,,," << "\n";
        f_log.close();

        // Release memory for this input tensor
        std::vector<std::vector<std::vector<float>>> empty;
        input_tensor.swap(empty);
    }

    // Calculate prediction errors for training data
    predictive_errors.clear();
    for (size_t i = 0; i < trainY.size(); i++) {
        float error = std::abs(trainY[i] - predicted_vals[i]);
        predictive_errors.push_back(error);
    }

    // Train generator
    generator->clear_update_state();
    generator->train(config.epoch_train, config.lr_train, predictive_errors);

    // Save after initial training if enabled
    if (config.save_enabled) {
        try {
            save_model();
            update_count = 0; // Reset counter after saving
        } catch (const std::exception &e) {
            std::cerr << "Failed to save initial model state: " << e.what() << std::endl;
        }
    }

    // Keep only the most recent lookback_len values in observed_vals
    if (observed_vals.size() > predictor_config.lookback_len) {
        std::vector<float> recent_vals(observed_vals.end() - predictor_config.lookback_len,
                                       observed_vals.end());
        observed_vals.swap(recent_vals); // Using swap for efficiency

        // Force memory release
        std::vector<float>(observed_vals).swap(observed_vals);
    }
}


float AdapAD::simplify_error(const std::vector<float> &errors, float N_sigma) {
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
    float sq_sum = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0f);
    float std_dev = std::sqrt(sq_sum / errors.size() - mean * mean);

    return mean + N_sigma * std_dev;
}

void AdapAD::save_model() {
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
        DIR *dir = opendir(config.save_path.c_str());
        if (dir != nullptr) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                // Check if file is a previous save for this parameter
                if (filename.find(parameter_name + "_model_") == 0 &&
                    filename.find(".bin") != std::string::npos) {
                    std::string old_file = config.save_path + "/" + filename;
                    if (remove(old_file.c_str()) != 0) {
                        std::cerr << "Warning: Could not remove old model file: " << old_file
                                  << std::endl;
                    } else {
                        std::cout << "Updated model save file for parameter: " << parameter_name
                                  << std::endl;
                    }
                }
            }
            closedir(dir);
        }

        // Create new file path with parameter name and timestamp
        std::string save_file =
            config.save_path + "/" + parameter_name + "_model_" + timestamp.str() + ".bin";

        std::ofstream file(save_file, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + save_file);
        }

        // Save metadata
        file.write(reinterpret_cast<const char *>(&minimal_threshold), sizeof(float));
        file.write(reinterpret_cast<const char *>(&value_range_config.lower_bound), sizeof(float));
        file.write(reinterpret_cast<const char *>(&value_range_config.upper_bound), sizeof(float));

        // Save critical configuration parameters
        file.write(reinterpret_cast<const char *>(&config.LSTM_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(&config.LSTM_size_layer), sizeof(int));
        file.write(reinterpret_cast<const char *>(&config.lookback_len), sizeof(int));
        file.write(reinterpret_cast<const char *>(&config.prediction_len), sizeof(int));

        // Save non-critical configuration parameters
        file.write(reinterpret_cast<const char *>(&config.epoch_train), sizeof(int));
        file.write(reinterpret_cast<const char *>(&config.epoch_update), sizeof(int));
        file.write(reinterpret_cast<const char *>(&config.epoch_update_generator), sizeof(int));
        file.write(reinterpret_cast<const char *>(&config.lr_train), sizeof(float));
        file.write(reinterpret_cast<const char *>(&config.lr_update), sizeof(float));
        file.write(reinterpret_cast<const char *>(&config.lr_update_generator), sizeof(float));
        file.write(reinterpret_cast<const char *>(&config.threshold_multiplier), sizeof(float));

        // Save predictor weights and biases directly to file
        data_predictor->save_weights(file);
        data_predictor->save_biases(file);

        // Save generator weights and biases directly to file
        generator->save_weights(file);
        generator->save_biases(file);

        // Ensure file is flushed and closed properly
        file.flush();
        file.close();

        // Force cleanup after saving
        data_predictor->clear_update_state();
        generator->clear_update_state();

    } catch (const std::exception &e) {
        std::cerr << "Error saving model state: " << e.what() << std::endl;
        throw;
    }
}

void AdapAD::load_model(const std::string &timestamp, const std::vector<float> &initial_data) {
    try {
        if (initial_data.size() < predictor_config.lookback_len) {
            throw std::runtime_error("Not enough initial data points provided. Need at least " +
                                     std::to_string(predictor_config.lookback_len) + " points.");
        }

        std::string load_file =
            config.save_path + "/" + parameter_name + "_model_" + timestamp + ".bin";

        if (access(load_file.c_str(), F_OK) == -1) {
            throw std::runtime_error("Model file does not exist: " + load_file);
        }

        std::ifstream file(load_file, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + load_file);
        }

        try {
            std::cout << "Loading metadata..." << std::endl;

            if (!file.good()) {
                throw std::runtime_error("Failed to read metadata");
            }

            float saved_minimal_threshold, saved_lower_bound, saved_upper_bound;
            file.read(reinterpret_cast<char *>(&saved_minimal_threshold), sizeof(float));
            file.read(reinterpret_cast<char *>(&saved_lower_bound), sizeof(float));
            file.read(reinterpret_cast<char *>(&saved_upper_bound), sizeof(float));

            // Load critical configuration parameters
            int saved_lstm_size, saved_lstm_layers, saved_lookback_len, saved_prediction_len;
            file.read(reinterpret_cast<char *>(&saved_lstm_size), sizeof(int));
            file.read(reinterpret_cast<char *>(&saved_lstm_layers), sizeof(int));
            file.read(reinterpret_cast<char *>(&saved_lookback_len), sizeof(int));
            file.read(reinterpret_cast<char *>(&saved_prediction_len), sizeof(int));

            // Check for critical configuration changes
            if (saved_lstm_size != config.LSTM_size ||
                saved_lstm_layers != config.LSTM_size_layer ||
                saved_lookback_len != config.lookback_len ||
                saved_prediction_len != config.prediction_len) {
                
                std::cerr << "Critical model configuration has changed. Cannot load model." << std::endl;
                std::cerr << "Saved model: LSTM_size=" << saved_lstm_size 
                          << ", LSTM_layers=" << saved_lstm_layers
                          << ", lookback_len=" << saved_lookback_len
                          << ", prediction_len=" << saved_prediction_len << std::endl;
                std::cerr << "Current config: LSTM_size=" << config.LSTM_size 
                          << ", LSTM_layers=" << config.LSTM_size_layer
                          << ", lookback_len=" << config.lookback_len
                          << ", prediction_len=" << config.prediction_len << std::endl;
                          
                throw std::runtime_error("Critical model configuration has changed");
            }

            // Load non-critical configuration parameters
            int saved_epoch_train, saved_epoch_update, saved_epoch_update_generator;
            float saved_lr_train, saved_lr_update, saved_lr_update_generator, saved_threshold_multiplier;
            file.read(reinterpret_cast<char *>(&saved_epoch_train), sizeof(int));
            file.read(reinterpret_cast<char *>(&saved_epoch_update), sizeof(int));
            file.read(reinterpret_cast<char *>(&saved_epoch_update_generator), sizeof(int));
            file.read(reinterpret_cast<char *>(&saved_lr_train), sizeof(float));
            file.read(reinterpret_cast<char *>(&saved_lr_update), sizeof(float));
            file.read(reinterpret_cast<char *>(&saved_lr_update_generator), sizeof(float));
            file.read(reinterpret_cast<char *>(&saved_threshold_multiplier), sizeof(float));
            

            // Log non-critical configuration changes
            if (saved_epoch_train != config.epoch_train) {
                std::cout << "Config updated: training.epochs.train changed from " 
                          << saved_epoch_train << " to " << config.epoch_train << std::endl;
            }
            if (saved_epoch_update != config.epoch_update) {
                std::cout << "Config updated: training.epochs.update changed from " 
                          << saved_epoch_update << " to " << config.epoch_update << std::endl;
            }
            if (saved_epoch_update_generator != config.epoch_update_generator) {
                std::cout << "Config updated: training.epochs.update_generator changed from " 
                          << saved_epoch_update_generator << " to " << config.epoch_update_generator << std::endl;
            }
            if (saved_lr_train != config.lr_train) {
                std::cout << "Config updated: training.learning_rates.train changed from " 
                          << saved_lr_train << " to " << config.lr_train << std::endl;
            }
            if (saved_lr_update != config.lr_update) {
                std::cout << "Config updated: training.learning_rates.update changed from " 
                          << saved_lr_update << " to " << config.lr_update << std::endl;
            }
            if (saved_lr_update_generator != config.lr_update_generator) {
                std::cout << "Config updated: training.learning_rates.update_generator changed from " 
                          << saved_lr_update_generator << " to " << config.lr_update_generator << std::endl;
            }
            if (saved_threshold_multiplier != config.threshold_multiplier) {
                std::cout << "Config updated: anomaly_detection.threshold_multiplier changed from " 
                          << saved_threshold_multiplier << " to " << config.threshold_multiplier << std::endl;
            }
            if (saved_minimal_threshold != minimal_threshold) {
                std::cout << "Config updated: anomaly_detection.minimal_threshold changed from " 
                          << saved_minimal_threshold << " to " << minimal_threshold << std::endl;
            }
            if (saved_lower_bound != value_range_config.lower_bound) {
                std::cout << "Config updated: anomaly_detection.value_range.lower changed from " 
                          << saved_lower_bound << " to " << value_range_config.lower_bound << std::endl;
            }
            if (saved_upper_bound != value_range_config.upper_bound) {
                std::cout << "Config updated: anomaly_detection.value_range.upper changed from " 
                          << saved_upper_bound << " to " << value_range_config.upper_bound << std::endl;
            }

            std::cout << "Initializing layer caches." << std::endl;
            // Initialize layer caches
            data_predictor->initialize_layer_cache();
            generator->initialize_layer_cache();

            std::cout << "Loading weights and biases." << std::endl;
            // Load weights and biases
            try {
                data_predictor->load_weights(file);
                data_predictor->load_biases(file);
                generator->load_weights(file);
                generator->load_biases(file);
            } catch (const std::exception &e) {
                throw std::runtime_error("Failed to load weights/biases: " + std::string(e.what()));
            }

            data_predictor->clear_update_state();
            generator->clear_update_state();

            std::cout << "Initializing observed values." << std::endl;
            // Initialize observed_vals with exactly lookback_len points
            observed_vals.clear();
            for (size_t i = 0; i < predictor_config.lookback_len; i++) {
                float normalized = normalize_data(initial_data[i]);
                observed_vals.push_back(normalized);
            }

            // Initialize other vectors with minimal size
            predicted_vals.clear();
            predictive_errors.clear();
            thresholds.clear();

            // Force memory release
            std::vector<float> empty_vec;
            predicted_vals.swap(empty_vec);
            empty_vec.clear();
            predictive_errors.swap(empty_vec);
            empty_vec.clear();
            thresholds.swap(empty_vec);

            // Properly close the file
            file.close();

            std::cout << "Successfully loaded model state for " << parameter_name << std::endl;

        } catch (const std::runtime_error &e) {
            file.close();
            throw std::runtime_error("Error during model loading: " + std::string(e.what()));
        }

    } catch (const std::exception &e) {
        std::cerr << "Error loading model state: " << e.what() << std::endl;
        throw;
    }
}

std::string AdapAD::get_state_filename() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << config.save_path << "model_state_"
       << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << ".bin";
    return ss.str();
}

void AdapAD::clean_old_saves(size_t keep_count) {
    try {
        std::vector<std::string> files;
        DIR *dir = opendir(config.save_path.c_str());
        if (dir != nullptr) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".bin") {
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
    } catch (const std::exception &e) {
        std::cerr << "Failed to clean old saves: " << e.what() << std::endl;
    }
}

bool AdapAD::has_saved_model() const {
    DIR *dir = opendir(config.save_path.c_str());
    if (dir == nullptr) {
        return false;
    }

    bool found = false;
    struct dirent *entry;
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

void AdapAD::load_latest_model(const std::vector<float> &initial_data) {
    DIR *dir = opendir(config.save_path.c_str());
    if (dir == nullptr) {
        throw std::runtime_error("Could not open save directory");
    }

    std::string latest_file;
    struct dirent *entry;
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
        load_model(timestamp, initial_data);
    } else {
        throw std::runtime_error("No saved model found for " + parameter_name);
    }
}
