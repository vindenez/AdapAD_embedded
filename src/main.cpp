#include "adapad.hpp"
#include "config.hpp"
#include "yaml_handler.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <memory>
#include <sys/resource.h>

struct DataPoint {
    float value;
    bool is_anomaly;
};

// Helper function to read CSV column
std::vector<DataPoint> read_csv_column(const std::string& filename, int column_index) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        
        // Skip to desired column
        for (int i = 0; i <= column_index; i++) {
            getline(ss, field, ',');
        }
        
        DataPoint point;
        point.value = std::stof(field);
        point.is_anomaly = false;
        data.push_back(point);
    }
    return data;
}

// Helper function to get current memory usage
size_t get_memory_usage() {
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return (size_t)rusage.ru_maxrss;
}

int main() {
    std::chrono::high_resolution_clock::time_point total_start_time = 
        std::chrono::high_resolution_clock::now();
    
    // Get parameters from config
    Config& config = Config::getInstance();
    
    // Load the config file
    if (!config.load("config.yaml")) {
        std::cerr << "Failed to load config.yaml" << std::endl;
        return 1;
    }
    
    // Read CSV header to get parameter names and order
    std::ifstream file(config.data_source_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data file: " << config.data_source_path << std::endl;
        return 1;
    }
    
    std::string header;
    std::getline(file, header);
    file.close();
    
    // Parse header into parameter names
    std::vector<std::string> csv_parameters;
    std::stringstream ss(header);
    std::string param;
    
    // Skip timestamp
    std::getline(ss, param, ',');
    
    // Get remaining parameters
    while (std::getline(ss, param, ',')) {
        csv_parameters.push_back(param);
    }
    
    std::cout << "Found " << csv_parameters.size() << " parameters in CSV:" << std::endl;
    for (const auto& param : csv_parameters) {
        std::cout << "- " << param << std::endl;
    }
    
    if (csv_parameters.empty()) {
        std::cerr << "Error: No parameters found in CSV header" << std::endl;
        return 1;
    }
    
    // Initialize models and measure memory usage
    std::vector<std::unique_ptr<AdapAD>> models;
    size_t initial_memory = get_memory_usage();

    auto predictor_config = init_predictor_config();  // Get predictor config once

        // Debug print save settings
    std::cout << "\nModel save settings:" << std::endl;
    std::cout << "Save enabled: " << config.save_enabled << std::endl;
    std::cout << "Save interval: " << config.save_interval << std::endl;
    std::cout << "Save path: " << config.save_path << std::endl;

    std::cout << "\nInitializing models..." << std::endl;
    for (size_t i = 0; i < csv_parameters.size(); ++i) {
        const std::string& param_name = csv_parameters[i];
        
        // Check if parameter exists in config
        std::string config_key = "data.parameters.Austevoll_nord." + param_name + ".minimal_threshold";
        bool param_configured = false;
        
        for (const auto& pair : config.get_config_map()) {
            if (pair.first == config_key) {
                param_configured = true;
                break;
            }
        }
        
        if (!param_configured) {
            std::cout << "Warning: Parameter '" << param_name 
                      << "' not configured in config.yaml, skipping..." << std::endl;
            continue;
        }
        
        float minimal_threshold;
        auto value_range_config = init_value_range_config("data.parameters.Austevoll_nord." + param_name, minimal_threshold);
        
        if (minimal_threshold == 0.0f) {
            std::cerr << "Error: It is mandatory to set a minimal threshold in config.yaml for " 
                      << param_name << std::endl;
            continue;
        }
        
        models.push_back(std::unique_ptr<AdapAD>(new AdapAD(
            predictor_config, value_range_config, minimal_threshold, param_name)));
    }

    size_t total_memory = get_memory_usage() - initial_memory;
    std::cout << "Total memory usage for all models: " << total_memory / 1024.0 << " MB" << std::endl;
    
    // Read training data for all models
    std::vector<std::vector<DataPoint>> all_data;
    for (size_t i = 0; i < models.size(); ++i) {
        all_data.push_back(read_csv_column(config.data_source_path, i + 1));
    }
    
    // Training phase
    std::cout << "\nStarting training phase..." << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < models.size(); ++i) {
        // Get initial data points for lookback
        std::vector<float> initial_data;
        for (size_t j = 0; j < predictor_config.lookback_len && j < all_data[i].size(); ++j) {
            initial_data.push_back(all_data[i][j].value);
        }

        if (models[i]->has_saved_model()) {
            std::cout << "Found saved model for " << csv_parameters[i] << ", loading..." << std::endl;
            try {
                models[i]->load_latest_model(initial_data);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load model for " << csv_parameters[i] << ": " << e.what() << std::endl;
                std::cout << "Falling back to training new model..." << std::endl;
                goto train_new_model;
            }
        } else {
            train_new_model:
            std::vector<float> training_data;
            for (size_t j = 0; j < predictor_config.train_size && j < all_data[i].size(); ++j) {
                training_data.push_back(all_data[i][j].value);
            }
            models[i]->set_training_data(training_data);
            models[i]->train();
        }
    }
    
    auto train_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_time = train_end - train_start;
    std::cout << "Training completed in " << train_time.count() << " seconds" << std::endl;
    
    // Online learning phase
    std::cout << "\nStarting online learning phase..." << std::endl;
    size_t total_predictions = 0;
    double total_processing_time = 0.0;
    
    // Process each time step
    double window_total_time = 0.0;
    size_t window_count = 0;
    const size_t WINDOW_SIZE = 5;

    // Initialize buffers for each model to accumulate initial data points
    std::vector<std::vector<float>> data_buffers(models.size());

    for (size_t t = predictor_config.train_size; t < all_data[0].size(); ++t) {
        std::cout << "Processing time step " << t << std::endl;
        double timestep_total = 0.0;
        
        // Process all models for this time step
        for (size_t i = 0; i < models.size(); ++i) {
            std::cout << "Processing model " << i << " (" << csv_parameters[i] << ")" << std::endl;
            auto model_start = std::chrono::high_resolution_clock::now();
            
            try {
                float measured_value = all_data[i][t].value;
                data_buffers[i].push_back(measured_value);
                
                // Only process if we have enough data points (lookback_len + 1)
                if (data_buffers[i].size() >= predictor_config.lookback_len + 1) {
                    std::cout << "Processing " << csv_parameters[i] 
                              << " with " << data_buffers[i].size() 
                              << " data points" << std::endl;
                    
                    bool is_anomalous = models[i]->is_anomalous(measured_value);
                    std::cout << "Anomaly check completed" << std::endl;
                    models[i]->clean();
                    all_data[i][t].is_anomaly = is_anomalous;
                    
                    // Keep only the most recent lookback_len + 1 points
                    if (data_buffers[i].size() > predictor_config.lookback_len + 1) {
                        data_buffers[i].erase(data_buffers[i].begin());
                    }
                } else {
                    std::cout << "Accumulating data for " << csv_parameters[i] 
                              << " (" << data_buffers[i].size() << "/"
                              << (predictor_config.lookback_len + 1) << " points)" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << csv_parameters[i] 
                          << " at time " << t << ": " << e.what() << std::endl;
            }
            
            auto model_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> model_time = model_end - model_start;
            timestep_total += model_time.count();
        }
        
        total_predictions++;
        total_processing_time += timestep_total;
        window_total_time += timestep_total;
        window_count++;
        
        // Calculate and print average when window is full
        if (window_count == WINDOW_SIZE) {
            double window_average = window_total_time / WINDOW_SIZE;
            std::cout << "Average processing time for last " << WINDOW_SIZE 
                      << " points (all models): " << window_average 
                      << " seconds" << std::endl;
            std::cout << "Average time per model: " 
                      << (window_average / models.size()) 
                      << " seconds" << std::endl;
            
            // Reset window counters
            window_total_time = 0.0;
            window_count = 0;
        }
        
        // Still keep the 1000-point progress updates
        if (t % 1000 == 0) {
            std::cout << "\nProcessed " << t << " time steps" << std::endl;
            std::cout << "Overall average processing time per time step (all models): " 
                      << (total_processing_time / total_predictions) 
                      << " seconds" << std::endl;
            std::cout << "Overall average time per model: " 
                      << (total_processing_time / total_predictions / models.size()) 
                      << " seconds\n" << std::endl;
        }
    }
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;
    
    // Print final statistics
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "Total processing time: " << total_elapsed.count() << " seconds" << std::endl;
    std::cout << "Overall average processing time per time step (all models): " 
              << (total_processing_time / total_predictions) << " seconds" << std::endl;
    std::cout << "Overall average time per model: " 
              << (total_processing_time / total_predictions / models.size()) 
              << " seconds" << std::endl;
    std::cout << "Memory usage: " << get_memory_usage() / 1024.0 << " MB" << std::endl;
    
    
    return 0;
} 