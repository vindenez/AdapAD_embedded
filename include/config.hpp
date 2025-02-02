#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <map>

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
    float lr_update;
};

// Configuration structure for value ranges
struct ValueRangeConfig {
    float lower_bound;     // Lower bound of sensor values
    float upper_bound;     // Upper bound of sensor values
};

class Config {
public:
    static Config& getInstance() {
        static Config instance;
        return instance;
    }

    bool load(const std::string& yaml_path);
    void apply_data_source_config();

    // Data paths
    std::string data_source;
    std::string data_source_path;
    std::string data_val_path;
    std::string log_file_path;

    // Training parameters
    int epoch_train;
    float lr_train;
    int epoch_update;
    float lr_update;
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
    unsigned int random_seed;
    bool verbose_output;

    // Model state configuration
    bool save_enabled;
    int save_interval;
    std::string save_path;

private:
    Config() {
        save_enabled = true;    // default value
        save_interval = 48;     // default value
        save_path = "model_states/";
    } // Private constructor for singleton
    std::map<std::string, std::string> config_map; // Store the parsed YAML configuration
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    // Helper methods for accessing config values
    float get_float(const std::string& key, float default_value = 0.0f);
    int get_int(const std::string& key, int default_value = 0);
    std::string get_string(const std::string& key, const std::string& default_value = "");
    bool get_bool(const std::string& key, bool default_value = false);
};

// Declare the configuration functions
PredictorConfig init_predictor_config();
ValueRangeConfig init_value_range_config(const std::string& data_source, float& minimal_threshold);

#endif // CONFIG_HPP
