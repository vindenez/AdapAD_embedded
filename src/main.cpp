#include "adapad.hpp"
#include "config.hpp"
#include "yaml_handler.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

struct DataPoint {
    float value;
    bool is_anomaly;
};

struct SystemStats {
    long voluntary_switches;
    long involuntary_switches;
    struct timeval user_time;
    struct timeval system_time;
};

struct CPUStats {
    unsigned long long user;
    unsigned long long nice;
    unsigned long long system;
    unsigned long long idle;
    unsigned long long iowait;
    unsigned long long irq;
    unsigned long long softirq;
    unsigned long long steal;
};

class DataStream {
  private:
    std::ifstream file;
    std::vector<std::string> headers;
    size_t current_line;
    size_t num_columns;
    std::string file_path;

  public:
    DataStream(const std::string &path) : file_path(path), current_line(0) {
        file.open(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open data file: " + path);
        }

        // Read header
        std::string header;
        std::getline(file, header);

        // Parse header
        std::stringstream ss(header);
        std::string col;
        while (std::getline(ss, col, ',')) {
            headers.push_back(col);
        }
        num_columns = headers.size();
    }

    std::vector<std::string> get_headers() const { return headers; }

    bool read_next_line(std::vector<float> &values) {
        if (!file.good())
            return false;

        std::string line;
        if (!std::getline(file, line))
            return false;

        std::stringstream ss(line);
        std::string field;
        values.clear();

        // Skip timestamp
        std::getline(ss, field, ',');

        // Read values
        while (std::getline(ss, field, ',')) {
            // Trim whitespace
            field.erase(0, field.find_first_not_of(" \t\r\n"));
            field.erase(field.find_last_not_of(" \t\r\n") + 1);

            try {
                if (field.empty() || field == "NA" || field == "NaN" || field == "-" ||
                    field == "0.0") {
                    values.push_back(-999.0f);
                } else {
                    values.push_back(std::stof(field));
                }
            } catch (const std::exception &e) {
                values.push_back(-999.0f);
            }
        }

        current_line++;
        return true;
    }

    void reset() {
        file.clear();
        file.seekg(0);
        std::string header;
        std::getline(file, header); // Skip header
        current_line = 0;
    }

    size_t get_current_line() const { return current_line; }

    size_t get_num_columns() const { return num_columns; }
};

std::vector<DataPoint> read_csv_column(const std::string &filename, int column_index) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;
    int line_number = 0;

    // Skip header
    std::getline(file, line);
    line_number++;

    while (std::getline(file, line)) {
        line_number++;
        std::stringstream ss(line);
        std::string field;

        // Skip to desired column
        for (int i = 0; i <= column_index; i++) {
            if (!getline(ss, field, ',')) {
                std::cerr << "Warning: Missing column at line " << line_number << std::endl;
                field = ""; // Ensure field is empty for missing columns
            }
        }

        DataPoint point;
        // Trim whitespace
        field.erase(0, field.find_first_not_of(" \t\r\n"));
        field.erase(field.find_last_not_of(" \t\r\n") + 1);

        // Check for empty or invalid values
        if (field.empty() || field == "NA" || field == "NaN" || field == "-" || field == "0.0") {
            std::cerr << "Warning: Invalid/missing value at line " << line_number << ", column "
                      << column_index << ": '" << field << "', setting to -999" << std::endl;
            point.value = -999.0f;
        } else {
            try {
                point.value = std::stof(field);
            } catch (const std::exception &e) {
                std::cerr << "Warning: Failed to parse value at line " << line_number << ", column "
                          << column_index << ": '" << field << "', setting to -999" << std::endl;
                point.value = -999.0f;
            }
        }

        point.is_anomaly = false;
        data.push_back(point);
    }

    if (data.empty()) {
        throw std::runtime_error("No data points found in column " + std::to_string(column_index));
    }

    // Count valid vs invalid values
    size_t invalid_count = std::count_if(data.begin(), data.end(),
                                         [](const DataPoint &p) { return p.value == -999.0f; });

    std::cout << "Column " << column_index << " statistics:" << std::endl;
    std::cout << "- Total points: " << data.size() << std::endl;
    std::cout << "- Valid points: " << (data.size() - invalid_count) << std::endl;
    std::cout << "- Missing/invalid points: " << invalid_count << std::endl;

    return data;
}

int get_cpu_freq() {
    std::ifstream freq_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    int freq = 0;
    if (freq_file) {
        freq_file >> freq;
    }
    return freq;
}

float get_cpu_temp() {
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    int temp = 0;
    if (temp_file) {
        temp_file >> temp;
    }
    return temp / 1000.0f; // millicelsius to celsius
}

SystemStats get_system_stats() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return {usage.ru_nvcsw, usage.ru_nivcsw, usage.ru_utime, usage.ru_stime};
}

CPUStats get_cpu_stats() {
    std::ifstream stat_file("/proc/stat");
    CPUStats stats{0};
    std::string cpu_label;
    if (stat_file) {
        stat_file >> cpu_label >> stats.user >> stats.nice >> stats.system >> stats.idle >>
            stats.iowait >> stats.irq >> stats.softirq >> stats.steal;
    }
    return stats;
}

double calculate_cpu_usage(const CPUStats &start, const CPUStats &end) {
    unsigned long long start_idle = start.idle + start.iowait;
    unsigned long long end_idle = end.idle + end.iowait;

    unsigned long long start_total = start.user + start.nice + start.system + start.idle +
                                     start.iowait + start.irq + start.softirq + start.steal;
    unsigned long long end_total = end.user + end.nice + end.system + end.idle + end.iowait +
                                   end.irq + end.softirq + end.steal;

    unsigned long long total_diff = end_total - start_total;
    unsigned long long idle_diff = end_idle - start_idle;

    if (total_diff == 0)
        return 0.0;
    return (100.0 * (total_diff - idle_diff)) / total_diff;
}

int main() {
    std::chrono::high_resolution_clock::time_point total_start_time =
        std::chrono::high_resolution_clock::now();

    // Get parameters from config
    Config &config = Config::getInstance();

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
    for (const auto &param : csv_parameters) {
        std::cout << "- " << param << std::endl;
    }

    if (csv_parameters.empty()) {
        std::cerr << "Error: No parameters found in CSV header" << std::endl;
        return 1;
    }

    // Reserve space for parameters to avoid reallocations
    csv_parameters.reserve(15); // Adjust based on expected parameter count

    // Initialize models
    std::vector<std::unique_ptr<AdapAD>> models;

    auto predictor_config = init_predictor_config(); // Get predictor config once

    // Debug print save settings
    std::cout << "\nModel save settings:" << std::endl;
    std::cout << "Save enabled: " << config.save_enabled << std::endl;
    std::cout << "Load enabled: " << config.load_enabled << std::endl;
    std::cout << "Save interval: " << config.save_interval << std::endl;
    std::cout << "Save path: " << config.save_path << std::endl;

    // Pre-allocate models vector
    models.reserve(csv_parameters.size());

    // Cache config map to avoid repeated lookups
    const auto &config_map = config.get_config_map();

    std::cout << "\nInitializing models..." << std::endl;
    for (size_t i = 0; i < csv_parameters.size(); ++i) {
        const std::string &param_name = csv_parameters[i];

        // Construct config key once
        const std::string config_key =
            "data.parameters." + config.data_source + "." + param_name + ".minimal_threshold";

        // Use find instead of iteration
        if (config_map.find(config_key) == config_map.end()) {
            std::cout << "Warning: Parameter '" << param_name
                      << "' not configured in config.yaml, skipping..." << std::endl;
            continue;
        }

        float minimal_threshold;
        auto value_range_config = init_value_range_config(
            "data.parameters." + config.data_source + "." + param_name, minimal_threshold);

        if (minimal_threshold == 0.0f) {
            std::cerr << "Error: It is mandatory to set a minimal threshold in "
                         "config.yaml for "
                      << param_name << std::endl;
            continue;
        }

        models.push_back(std::unique_ptr<AdapAD>(
            new AdapAD(predictor_config, value_range_config, minimal_threshold, param_name)));
    }

    // Create data stream
    DataStream data_stream(config.data_source_path);

    // Training phase
    std::cout << "\nStarting training phase..." << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();

    // Read training data in chunks
    std::vector<float> line_values;
    std::vector<std::vector<float>> training_data(models.size());

    // Read training data
    for (size_t j = 0; j < predictor_config.train_size; ++j) {
        if (!data_stream.read_next_line(line_values)) {
            throw std::runtime_error("Not enough training data");
        }

        for (size_t i = 0; i < models.size(); ++i) {
            if (i < line_values.size()) {
                training_data[i].push_back(line_values[i]);
            }
        }
    }

    // Train models
    for (size_t i = 0; i < models.size(); ++i) {
        if (config.load_enabled && models[i]->has_saved_model()) {
            std::cout << "Found saved model for " << csv_parameters[i] << ", loading..."
                      << std::endl;
            try {
                models[i]->load_latest_model(training_data[i]);
            } catch (const std::exception &e) {
                std::cerr << "Failed to load model for " << csv_parameters[i] << ": " << e.what()
                          << std::endl;
                std::cout << "Falling back to training new model..." << std::endl;
                models[i]->set_training_data(training_data[i]);
                models[i]->train();
            }
        } else {
            models[i]->set_training_data(training_data[i]);
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

    while (data_stream.read_next_line(line_values)) {
        int freq_before = get_cpu_freq();
        float temp_before = get_cpu_temp();
        auto start_time = std::chrono::high_resolution_clock::now();
        auto cpu_stats_before = get_cpu_stats();

        std::cout << "\nTimestep " << data_stream.get_current_line()
                  << " (CPU Freq: " << freq_before / 1000 << " MHz"
                  << ", Temp: " << temp_before << "°C)" << std::endl;

        double timestep_total = 0.0;

        // Process all models for this timestep
        for (size_t i = 0; i < models.size(); ++i) {
            auto model_start = std::chrono::high_resolution_clock::now();
            auto stats_before = get_system_stats();

            try {
                // Process single new value
                const float measured_value = (i < line_values.size()) ? line_values[i] : -999.0f;
                bool is_anomaly = models[i]->is_anomalous(measured_value);

                auto model_end = std::chrono::high_resolution_clock::now();
                double model_time = std::chrono::duration<double>(model_end - model_start).count();

                timestep_total += model_time;

                // Enhanced embedded system logging
                auto stats_after = get_system_stats();
                double system_time =
                    (stats_after.system_time.tv_sec - stats_before.system_time.tv_sec) +
                    (stats_after.system_time.tv_usec - stats_before.system_time.tv_usec) / 1e6;

                std::cout << " (Sys=" << system_time << "s"
                          << ", CSw="
                          << (stats_after.voluntary_switches - stats_before.voluntary_switches)
                          << "v/"
                          << (stats_after.involuntary_switches - stats_before.involuntary_switches)
                          << "i)" << std::endl;

            } catch (const std::exception &e) {
                std::cerr << "Error processing " << csv_parameters[i] << " at time "
                          << data_stream.get_current_line() << ": " << e.what() << std::endl;
            }
        }

        float temp_after = get_cpu_temp();
        int freq_after = get_cpu_freq();
        auto cpu_stats_after = get_cpu_stats();
        double cpu_usage = calculate_cpu_usage(cpu_stats_before, cpu_stats_after);

        std::cout << "\nTimestep Summary:" << std::endl;
        std::cout << "- Total time: " << timestep_total << "s" << std::endl;
        std::cout << "- CPU Freq: " << freq_after / 1000 << "MHz (Δ"
                  << (freq_after - freq_before) / 1000 << "MHz)" << std::endl;
        std::cout << "- CPU Temp: " << temp_after << "°C (Δ" << (temp_after - temp_before) << "°C)"
                  << std::endl;
        std::cout << "- CPU Usage: " << cpu_usage << "%" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

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
              << (total_processing_time / total_predictions / models.size()) << " seconds"
              << std::endl;

    return 0;
}