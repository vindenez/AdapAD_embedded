#include "config.hpp"
#include <iostream>

// Global Configuration Variables
std::string data_source_path = "../data/Tide_pressure.validation_stage.csv";
std::string data_source = "Tide_pressure";

int epoch_train = 3000;
float lr_train = 0.00005;
int epoch_update = 100;
float lr_update = 0.00005;
int update_G_epoch = 100;
float update_G_lr = 0.00005;
int LSTM_size = 100;
int LSTM_size_layer = 3;

// Initialize predictor configuration
PredictorConfig init_predictor_config() {
    PredictorConfig predictor_config;
    predictor_config.lookback_len = 3;
    predictor_config.prediction_len = 1;
    predictor_config.train_size = 5 * predictor_config.lookback_len + predictor_config.prediction_len;
    predictor_config.num_layers = LSTM_size_layer;
    predictor_config.hidden_size = LSTM_size;
    predictor_config.num_classes = 1;  // Assuming a single output for prediction
    predictor_config.input_size = predictor_config.lookback_len;  // Assuming input size matches lookback length
    predictor_config.epoch_update = 100;
    predictor_config.lr_update = 0.00005f;

    return predictor_config;
}

// Initialize value range configuration based on data source
ValueRangeConfig init_value_range_config(const std::string& data_source, float& minimal_threshold) {
    ValueRangeConfig value_range_config;

    if (data_source == "Tide_pressure") {
        value_range_config.lower_bound = 713;
        value_range_config.upper_bound = 763;
        minimal_threshold = 0.0038;

        update_G_epoch = 5;
        update_G_lr = 0.00005;
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

    return value_range_config;
}
