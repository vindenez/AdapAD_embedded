#ifndef CONFIG_HPP
#define CONFIG_HPP

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

// Declare the configuration functions
PredictorConfig init_predictor_config();
ValueRangeConfig init_value_range_config(const std::string& data_source, float& minimal_threshold);

// Declare global configuration variables
extern std::string data_source_path;
extern std::string data_source;

extern int epoch_train;
extern float lr_train;
extern int epoch_update;
extern float lr_update;
extern int update_G_epoch;
extern float update_G_lr;
extern int LSTM_size;
extern int LSTM_size_layer;

#endif // CONFIG_HPP
