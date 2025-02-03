#include "adapad.hpp"
#include "config.hpp"
#include "yaml_handler.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>

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
    
    // Process each parameter sequentially
    for (size_t i = 0; i < csv_parameters.size(); ++i) {
        const std::string& param_name = csv_parameters[i];
        int column_index = i + 1;  // +1 to skip timestamp column
        
        std::cout << "\nProcessing parameter: " << param_name << std::endl;
        
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
        
        auto predictor_config = init_predictor_config();
        float minimal_threshold;
        auto value_range_config = init_value_range_config("data.parameters.Austevoll_nord." + param_name, minimal_threshold);
        
        if (minimal_threshold == 0.0f) {
            std::cerr << "Error: It is mandatory to set a minimal threshold in config.yaml for " 
                      << param_name << std::endl;
            continue;
        }
        
        // Read data for this parameter
        std::vector<DataPoint> parameter_data = read_csv_column(config.data_source_path, column_index);
        
        // Initialize AdapAD for this parameter with its specific name
        AdapAD adapad(predictor_config, value_range_config, minimal_threshold, param_name);
        
        std::cout << "GATHERING DATA FOR TRAINING " << param_name << "... " 
                  << predictor_config.train_size << std::endl;
        
        std::vector<float> observed_data;
        size_t total_decisions = 0;
        
        auto param_start_time = std::chrono::high_resolution_clock::now();
        
        // Process data points
        for (size_t j = 0; j < parameter_data.size(); ++j) {
            float measured_value = parameter_data[j].value;
            observed_data.push_back(measured_value);
            
            if (observed_data.size() == predictor_config.train_size) {
                adapad.set_training_data(observed_data);
                adapad.train();
                std::cout << "\n------------STARTING ONLINE LEARNING PHASE FOR " 
                          << param_name << "------------" << std::endl;
                
                param_start_time = std::chrono::high_resolution_clock::now();
            }
            else if (observed_data.size() > predictor_config.train_size) {
                try {
                    if (observed_data.size() < predictor_config.lookback_len + 1) {
                        std::cerr << "Not enough data for prediction" << std::endl;
                        continue;
                    }
                    
                    bool is_anomalous = adapad.is_anomalous(measured_value);
                    adapad.clean();
                    total_decisions++;
                } catch (const std::exception& e) {
                    std::cerr << "Error in online learning phase for " << param_name 
                              << ": " << e.what() << std::endl;
                    continue;
                }
            }
            else {
                std::cout << observed_data.size() << "/" << predictor_config.train_size 
                          << " to warmup training for " << param_name << std::endl;
            }
        }
        
        auto param_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> param_elapsed = param_end_time - param_start_time;
        
        std::cout << "Processed " << total_decisions << " points for " << param_name 
                  << " in " << param_elapsed.count() << " seconds" << std::endl;
        std::cout << "Average time per decision: " 
                  << (param_elapsed.count() / total_decisions) << " seconds" << std::endl;
        
        std::cout << "Results for " << param_name << " at " << adapad.get_log_filename() << std::endl;
    }
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;
    std::cout << "\nTotal processing time for all parameters: " 
              << total_elapsed.count() << " seconds" << std::endl;
    
    return 0;
} 