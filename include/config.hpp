#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <map>
#include <vector>
#include "yaml_handler.hpp"
#include <algorithm> 
// Configuration structure for predictor settings
struct PredictorConfig {
    int lookback_len;      // Lookback length for LSTM
    int prediction_len;    // Prediction length
    int train_size;        // Size of the training set
    int num_layers;        // Number of LSTM layers
    int hidden_size;       // Hidden size for LSTM
    int num_classes;       // Number of output classes (usually 1 for regression tasks)
    int input_size;        // Input size for LSTM, can be set to lookback_len
    int epoch_train;
    float lr_train;
    int epoch_update;
    int epoch_update_generator;    
    float lr_update;
    float lr_update_generator;     
};

// Configuration structure for value ranges
struct ValueRangeConfig {
    float lower_bound;     // Lower bound of sensor values
    float upper_bound;     // Upper bound of sensor values
};

class Config {
private:
    Config() {
        save_enabled = false;
        load_enabled = false;   
        save_interval = 48;    
        save_path = "model_states/";
    }
    std::map<std::string, std::string> config_map;
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    // Helper methods for accessing config values
    float get_float(const std::string& key, float default_value = 0.0f);
    int get_int(const std::string& key, int default_value = 0);
    std::string get_string(const std::string& key, const std::string& default_value = "");
    bool get_bool(const std::string& key, bool default_value = false);

public:
    static Config& getInstance() {
        static Config instance;
        return instance;
    }

    bool load(const std::string& yaml_path);
    void apply_data_source_config();

    // Get list of parameters for a given source
    std::vector<std::string> get_parameters(const std::string& source) const {
        std::vector<std::string> params;
        std::string base_key = "data.parameters." + source;
        
        // Iterate through config_map to find matching parameters
        for (const auto& pair : config_map) {
            const std::string& key = pair.first;
            if (key.compare(0, base_key.length(), base_key) == 0) {
                size_t pos = key.find('.', base_key.length() + 1);
                if (pos != std::string::npos) {
                    std::string param = key.substr(base_key.length() + 1, 
                                                 pos - base_key.length() - 1);
                    if (std::find(params.begin(), params.end(), param) == params.end()) {
                        params.push_back(param);
                    }
                }
            }
        }
        
        return params;
    }

    // Data paths
    std::string data_source;
    std::string data_source_path;
    std::string data_val_path;
    std::string log_file_path;

    // Training parameters
    int epoch_train;
    float lr_train;
    int epoch_update;
    int epoch_update_generator;    
    float lr_update;
    float lr_update_generator;     
    int update_G_epoch;
    float update_G_lr;

    // Model architecture
    int LSTM_size;
    int LSTM_size_layer;
    int lookback_len;
    int prediction_len;
    int train_size;
    int num_classes;
    int input_size;

    // Anomaly detection
    float minimal_threshold;
    float threshold_multiplier;

    // Data preprocessing
    float lower_bound;
    float upper_bound;

    // System
    bool use_neon;
    bool use_16bit;
    unsigned int random_seed;
    // Model state configuration
    bool save_enabled;
    bool load_enabled;
    int save_interval;
    std::string save_path;

    // Public method to access config map
    const std::map<std::string, std::string>& get_config_map() const {
        return config_map;
    }
};

PredictorConfig init_predictor_config();
ValueRangeConfig init_value_range_config(const std::string& data_source, float& minimal_threshold);

#endif // CONFIG_HPP
