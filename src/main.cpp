#include "adapad.hpp"
#include "config.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

struct DataPoint {
    float value;
    bool is_anomaly;
};

// Helper function to read CSV file
std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp, value_str, anomaly_str;
        
        // Parse each column
        getline(ss, timestamp, ',');  // Skip timestamp
        getline(ss, value_str, ',');  // Get value
        getline(ss, anomaly_str, ','); // Get anomaly flag
        
        DataPoint point;
        point.value = std::stof(value_str);
        point.is_anomaly = (anomaly_str == "1");
        
        data.push_back(point);
    }
    return data;
}

float get_cpu_usage() { /* ... */ }
float get_power_usage() { /* ... */ }

int main() {
    // Initialize configurations
    float minimal_threshold;
    auto predictor_config = init_predictor_config();
    auto value_range_config = init_value_range_config(config::data_source, minimal_threshold);
    
    if (minimal_threshold == 0.0f) {
        std::cerr << "Error: It is mandatory to set a minimal threshold" << std::endl;
        return 1;
    }
    
    // Read the complete dataset
    std::vector<DataPoint> all_data = read_csv(config::data_source_path);
    
    // Initialize AdapAD
    AdapAD adapad(predictor_config, value_range_config, minimal_threshold);
    std::cout << "GATHERING DATA FOR TRAINING... " << predictor_config.train_size << std::endl;
    
    std::vector<float> observed_data;
    size_t total_decisions = 0;
    
    // Open performance log file
    std::ofstream perf_log("performance_metrics.csv");
    perf_log << "timestamp,phase,cpu_usage,power_watts,processing_time_ms\n";
    
    for (size_t i = 0; i < all_data.size(); ++i) {
        float measured_value = all_data[i].value;
        observed_data.push_back(measured_value);
        
        if (observed_data.size() == predictor_config.train_size) {
            // Monitor initial training phase
            auto phase_start = std::chrono::high_resolution_clock::now();
            float cpu_before = get_cpu_usage();
            float power_before = get_power_usage();
            
            adapad.set_training_data(observed_data);
            adapad.train();
            
            auto phase_end = std::chrono::high_resolution_clock::now();
            float cpu_after = get_cpu_usage();
            float power_after = get_power_usage();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);
            
            perf_log << i << ",initial_training,"
                    << (cpu_after + cpu_before)/2 << ","
                    << (power_after + power_before)/2 << ","
                    << duration.count() << "\n";
            perf_log.flush();
            
            std::cout << "\n------------STARTING ONLINE LEARNING PHASE------------" << std::endl;
        }
        else if (observed_data.size() > predictor_config.train_size) {
            try {
                if (observed_data.size() < predictor_config.lookback_len + 1) {
                    continue;
                }
                
                // Monitor online learning phase
                auto phase_start = std::chrono::high_resolution_clock::now();
                float cpu_before = get_cpu_usage();
                float power_before = get_power_usage();
                
                bool is_anomalous = adapad.is_anomalous(measured_value);
                adapad.clean();
                
                auto phase_end = std::chrono::high_resolution_clock::now();
                float cpu_after = get_cpu_usage();
                float power_after = get_power_usage();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);
                
                perf_log << i << ",online_learning,"
                        << (cpu_after + cpu_before)/2 << ","
                        << (power_after + power_before)/2 << ","
                        << duration.count() << "\n";
                perf_log.flush();
                
                total_decisions++;
            } catch (const std::exception& e) {
                std::cerr << "Error in online learning phase: " << e.what() << std::endl;
                continue;
            }
        }
        else {
            std::cout << observed_data.size() << "/" << predictor_config.train_size 
                      << " to warmup training" << std::endl;
        }
    }
    
    std::cout << "Done! Check result at " << adapad.get_log_filename() << std::endl;
    return 0;
} 