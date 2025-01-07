#include "config.hpp"
#include <iostream>

namespace config {
    // Located in /data
    std::string data_source_path = "data/Tide_pressure.csv";
    std::string data_val_path = "data/Tide_pressure.validation_stage.csv";
    std::string data_source = "Tide_pressure";

    // Located on SD Card on the module /mnt/data
    // std::string data_source_path = "/mnt/sdcard/data/Tide_pressure.csv";
    // std::string data_val_path = "/mnt/sdcard/data/Tide_pressure.validation_stage.csv";
    // std::string data_source = "Tide_pressure";

    // Training parameters
    int epoch_train = 500;              
    float lr_train = 0.0002f;           
    int epoch_update = 100;              
    float lr_update = 0.00005f;          
    int update_G_epoch = 100;            
    float update_G_lr = 0.00005f;        

    // Model architecture
    int LSTM_size = 100;                 
    int LSTM_size_layer = 2;            
    int lookback_len = 3;                
    int prediction_len = 1;              

    int train_size = 2 * lookback_len + prediction_len;
    int num_classes = 1;
    int input_size = lookback_len;

    // Anomaly detection
    float minimal_threshold = 0.0038f;   
    float threshold_multiplier = 1.0f;

    // Data preprocessing
    float lower_bound = 713.0f;          
    float upper_bound = 763.0f;          

    // Random seed for reproducibility
    unsigned int random_seed = 42;

    // Logging and debugging
    bool verbose_output = true;
    std::string log_file_path;
}

// Initialize predictor configuration
PredictorConfig init_predictor_config() {
    PredictorConfig predictor_config;
    predictor_config.lookback_len = config::lookback_len;
    predictor_config.prediction_len = config::prediction_len;
    predictor_config.train_size = config::train_size;
    predictor_config.num_layers = config::LSTM_size_layer;
    predictor_config.hidden_size = config::LSTM_size;
    predictor_config.num_classes = config::num_classes;
    predictor_config.input_size = config::input_size;
    predictor_config.epoch_train = config::epoch_train;
    predictor_config.lr_train = config::lr_train;
    predictor_config.epoch_update = config::epoch_update;
    predictor_config.lr_update = config::lr_update;

    return predictor_config;
}

ValueRangeConfig init_value_range_config(const std::string& data_source, float& minimal_threshold) {
    ValueRangeConfig value_range_config;

    if (data_source == "Tide_pressure") {
        value_range_config.lower_bound = config::lower_bound;
        value_range_config.upper_bound = config::upper_bound;
        minimal_threshold = config::minimal_threshold;
        config::epoch_train = 40;
        config::epoch_update = 40;
        config::update_G_epoch = 5;
        config::update_G_lr = 0.0002f;
        config::lr_update = 0.0002f;
    } else if (data_source == "Wave_height") {
        value_range_config.lower_bound = 0;
        value_range_config.upper_bound = 15.2;
        minimal_threshold = 0.3;
    } else if (data_source == "Seawater_temperature") {
        value_range_config.lower_bound = 9;
        value_range_config.upper_bound = 33;
        minimal_threshold = 0.02;
    } else {
        std::cerr << "Unsupported data source! You need to set the hyperparameters manually." << std::endl;
    }

    // Logging
    config::log_file_path = "adapad_log_" + 
                               std::to_string(config::epoch_train) + "_" +
                               std::to_string(config::train_size) + "_" +
                               std::to_string(config::LSTM_size_layer) + "_" +
                               std::to_string(config::epoch_update) + "_" +
                               std::to_string(config::update_G_epoch) + "_" +
                               std::to_string(config::update_G_lr) + "lrG_" + 
                               std::to_string(config::lr_update) + "lr_" + ".csv";

    return value_range_config;
}
