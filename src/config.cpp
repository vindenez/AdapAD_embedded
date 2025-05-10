#include "config.hpp"
#include "yaml_handler.hpp"
#include <iostream>
#include <sstream>

float Config::get_float(const std::string &key, float default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? std::stof(it->second) : default_value;
}

int Config::get_int(const std::string &key, int default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? std::stoi(it->second) : default_value;
}

std::string Config::get_string(const std::string &key, const std::string &default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? it->second : default_value;
}

bool Config::get_bool(const std::string &key, bool default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? (it->second == "true") : default_value;
}

bool Config::load(const std::string &yaml_path) {
    try {
        config_map = YAMLHandler::parse(yaml_path);

        // Load data paths
        data_source = get_string("data.source");
        data_source_path = get_string("data.paths.training");
        data_val_path = get_string("data.paths.validation");
        log_file_path = get_string("data.paths.log");

        // Load model architecture
        LSTM_size = get_int("model.lstm.size", 100);
        LSTM_size_layer = get_int("model.lstm.layers", 2);
        lookback_len = get_int("model.lstm.lookback_len", 3);
        prediction_len = get_int("model.lstm.prediction_len", 1);

        // Load save settings
        save_enabled = get_bool("model.save_enabled", false);
        load_enabled = get_bool("model.load_enabled", false);
        save_interval = get_int("model.save_interval", 48);
        save_path = get_string("model.save_path", "model_states/");

        // Load training parameters
        epoch_train = get_int("training.epochs.train", 20);
        epoch_update = get_int("training.epochs.update", 30);
        epoch_update_generator = get_int("training.epochs.update_generator", 30);
        lr_train = get_float("training.learning_rates.train", 0.015f);
        lr_update = get_float("training.learning_rates.update", 0.015f);
        lr_update_generator = get_float("training.learning_rates.update_generator", 0.015f);

        // Load system settings
        random_seed = get_int("system.random_seed", 42);
        use_neon = get_bool("system.use_neon", true);
        // Load anomaly detection parameters
        threshold_multiplier = get_float("anomaly_detection.threshold_multiplier", 1.0f);

        data_source = get_string("data_source");

        // Apply data source specific configuration
        apply_data_source_config();

        // Derived values
        train_size = 2 * lookback_len + prediction_len;
        num_classes = 1;
        input_size = lookback_len;

        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

void Config::apply_data_source_config() {
    std::string prefix = "data.sources." + data_source + ".";

    // Override bounds
    lower_bound = get_float(prefix + "bounds.lower");
    upper_bound = get_float(prefix + "bounds.upper");
    minimal_threshold = get_float(prefix + "minimal_threshold");

    // Override training parameters if they exist
    if (config_map.find(prefix + "epochs.train") != config_map.end()) {
        epoch_train = get_int(prefix + "epochs.train");
        epoch_update = get_int(prefix + "epochs.update");
        update_G_epoch = get_int(prefix + "epochs.update_generator");
    }

    if (config_map.find(prefix + "learning_rates.update") != config_map.end()) {
        lr_update = get_float(prefix + "learning_rates.update");
        update_G_lr = get_float(prefix + "learning_rates.update_generator");
    }
}

PredictorConfig init_predictor_config() {
    const auto &config = Config::getInstance();

    PredictorConfig predictor_config;
    predictor_config.lookback_len = config.lookback_len;
    predictor_config.prediction_len = config.prediction_len;
    predictor_config.train_size = config.train_size;
    predictor_config.num_layers = config.LSTM_size_layer;
    predictor_config.hidden_size = config.LSTM_size;
    predictor_config.num_classes = config.num_classes;
    predictor_config.input_size = config.input_size;
    predictor_config.epoch_train = config.epoch_train;
    predictor_config.lr_train = config.lr_train;
    predictor_config.epoch_update = config.epoch_update;
    predictor_config.epoch_update_generator = config.epoch_update_generator;
    predictor_config.lr_update = config.lr_update;
    predictor_config.lr_update_generator = config.lr_update_generator;

    return predictor_config;
}

ValueRangeConfig init_value_range_config(const std::string &data_source, float &minimal_threshold) {
    ValueRangeConfig config;
    const Config &cfg = Config::getInstance();

    // Get bounds
    std::string lower_key = data_source + ".bounds.lower";
    std::string upper_key = data_source + ".bounds.upper";
    std::string threshold_key = data_source + ".minimal_threshold";

    config.lower_bound = std::stof(cfg.get_config_map().at(lower_key));
    config.upper_bound = std::stof(cfg.get_config_map().at(upper_key));
    minimal_threshold = std::stof(cfg.get_config_map().at(threshold_key));

    return config;
}
