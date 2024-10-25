#pragma once
#include <string>

// Configuration structure for predictor settings
struct PredictorConfig {
    int lookback_len;      // Lookback length for LSTM
    int prediction_len;    // Prediction length
    int train_size;        // Size of the training set
    int num_layers;        // Number of LSTM layers
    int hidden_size;       // Hidden size for LSTM
    int num_classes;       // Number of output classes (usually 1 for regression tasks)
    int input_size;        // Input size for LSTM, can be set to lookback_len
    int epoch_update;
    float lr_update;
};

// Configuration structure for value ranges
struct ValueRangeConfig {
    float lower_bound;     // Lower bound of sensor values
    float upper_bound;     // Upper bound of sensor values
};

namespace config {
    // General configuration
    extern std::string data_source_path;
    extern std::string data_source;

    // Training parameters
    extern int epoch_train;
    extern float lr_train;
    extern int epoch_update;
    extern float lr_update;
    extern int update_G_epoch;
    extern float update_G_lr;

    // Model architecture
    extern int LSTM_size;
    extern int LSTM_size_layer;
    extern int lookback_len;
    extern int prediction_len;
    extern int train_size;
    extern int num_classes;
    extern int input_size;

    // Anomaly detection
    extern float minimal_threshold;

    // Data preprocessing
    extern float lower_bound;
    extern float upper_bound;

    // Logging and debugging
    extern bool verbose_output;
    extern std::string log_file_path;

    // Performance tuning
    extern int batch_size;
    extern float dropout_rate;
    
    // Early stopping
    extern int patience;
    extern float min_delta;

    // Random seed for reproducibility
    extern unsigned int random_seed;
}

// Declare the configuration functions
PredictorConfig init_predictor_config();
ValueRangeConfig init_value_range_config(const std::string& data_source, float& minimal_threshold);
