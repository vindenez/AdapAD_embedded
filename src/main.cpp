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
#include <numeric>

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
    
    // Reserve space for parameters to avoid reallocations
    csv_parameters.reserve(100);  // Adjust based on expected parameter count
    
    // Initialize models and measure memory usage
    std::vector<std::unique_ptr<AdapAD>> models;
    size_t initial_memory = get_memory_usage();

    auto predictor_config = init_predictor_config();  // Get predictor config once

        // Debug print save settings
    std::cout << "\nModel save settings:" << std::endl;
    std::cout << "Save enabled: " << config.save_enabled << std::endl;
    std::cout << "Save interval: " << config.save_interval << std::endl;
    std::cout << "Save path: " << config.save_path << std::endl;

    // Pre-allocate models vector
    models.reserve(csv_parameters.size());
    
    // Cache config map to avoid repeated lookups
    const auto& config_map = config.get_config_map();
    
    std::cout << "\nInitializing models..." << std::endl;
    for (size_t i = 0; i < csv_parameters.size(); ++i) {
        const std::string& param_name = csv_parameters[i];
        
        // Construct config key once
        const std::string config_key = "data.parameters.Austevoll_nord." + param_name + ".minimal_threshold";
        
        // Use find instead of iteration
        if (config_map.find(config_key) == config_map.end()) {
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
    
    // Pre-allocate data vectors
    all_data.reserve(models.size());
    for (auto& data : all_data) {
        data.reserve(1000);  // Adjust based on expected data size
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
    
    // Online learning phase - process one value at a time
    const size_t data_size = all_data[0].size();
    for (size_t t = predictor_config.train_size; t < data_size; ++t) {
        double timestep_total = 0.0;
        
        for (size_t i = 0; i < models.size(); ++i) {
            auto model_start = std::chrono::high_resolution_clock::now();
            
            try {
                // Process single new value
                const float measured_value = all_data[i][t].value;
                all_data[i][t].is_anomaly = models[i]->is_anomalous(measured_value);
                models[i]->clean();
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << csv_parameters[i] 
                          << " at time " << t << ": " << e.what() << std::endl;
            }
            
            timestep_total += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - model_start).count();
        }
        
        std::cout << "Time step " << t << " total processing time: " 
                  << timestep_total << " seconds" << std::endl;
        
        total_predictions++;
        total_processing_time += timestep_total;
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