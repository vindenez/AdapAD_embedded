#include "config.hpp"
#include "yaml_handler.hpp"
#include <iostream>
#include <sstream>

float Config::get_float(const std::string& key, float default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? std::stof(it->second) : default_value;
}

int Config::get_int(const std::string& key, int default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? std::stoi(it->second) : default_value;
}

std::string Config::get_string(const std::string& key, const std::string& default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? it->second : default_value;
}

bool Config::get_bool(const std::string& key, bool default_value) {
    auto it = config_map.find(key);
    return it != config_map.end() ? (it->second == "true") : default_value;
}

bool Config::load(const std::string& yaml_path) {
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
        lookback_len = get_int("model.lstm.lookback", 3);
        prediction_len = get_int("model.lstm.prediction_len", 1);

        // Load save settings
        save_enabled = get_bool("model.save_enabled", false);
        save_interval = get_int("model.save_interval", 48);
        save_path = get_string("model.save_path", "model_states/");

        // Load optimizer type from model section
        optimizer_config.type = get_string("model.optimizer", "adam");
        
        // Load Adam config from optimizer section
        optimizer_config.adam.epochs.train = get_int("optimizer.adam.epochs.train", 50);
        optimizer_config.adam.epochs.update = get_int("optimizer.adam.epochs.update", 40);
        optimizer_config.adam.epochs.update_generator = get_int("optimizer.adam.epochs.update_generator", 5);
        
        optimizer_config.adam.learning_rates.train = get_float("optimizer.adam.learning_rates.train", 0.0002f);
        optimizer_config.adam.learning_rates.update = get_float("optimizer.adam.learning_rates.update", 0.00005f);
        optimizer_config.adam.learning_rates.update_generator = get_float("optimizer.adam.learning_rates.update_generator", 0.00005f);
        
        // Load SGD config from optimizer section
        optimizer_config.sgd.epochs.train = get_int("optimizer.sgd.epochs.train", 200);
        optimizer_config.sgd.epochs.update = get_int("optimizer.sgd.epochs.update", 60);
        optimizer_config.sgd.epochs.update_generator = get_int("optimizer.sgd.epochs.update_generator", 10);
        
        optimizer_config.sgd.learning_rates.train = get_float("optimizer.sgd.learning_rates.train", 0.002f);
        optimizer_config.sgd.learning_rates.update = get_float("optimizer.sgd.learning_rates.update", 0.0005f);
        optimizer_config.sgd.learning_rates.update_generator = get_float("optimizer.sgd.learning_rates.update_generator", 0.0005f);
        
        // Load new SGD parameters
        optimizer_config.sgd.momentum = get_float("optimizer.sgd.momentum", 0.9f);
        optimizer_config.sgd.weight_decay = get_float("optimizer.sgd.weight_decay", 0.0001f);

        // Load system settings
        random_seed = get_int("system.random_seed", 42);
        verbose_output = get_bool("system.verbose_output", true);

        // Load anomaly detection parameters
        threshold_multiplier = get_float("anomaly_detection.threshold_multiplier", 1.0f);

        // Apply data source specific configuration
        apply_data_source_config();

        // Derived values
        train_size = 2 * lookback_len + prediction_len;
        num_classes = 1;
        input_size = lookback_len;

        return true;
    } catch (const std::exception& e) {
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
    const auto& config = Config::getInstance();
    const auto& current_epochs = config.get_current_epochs();
    const auto& current_lr = config.get_current_learning_rates();
    
    PredictorConfig predictor_config;
    predictor_config.lookback_len = config.lookback_len;
    predictor_config.prediction_len = config.prediction_len;
    predictor_config.train_size = config.train_size;
    predictor_config.num_layers = config.LSTM_size_layer;
    predictor_config.hidden_size = config.LSTM_size;
    predictor_config.num_classes = config.num_classes;
    predictor_config.input_size = config.input_size;
    predictor_config.epoch_train = current_epochs.train;
    predictor_config.lr_train = current_lr.train;
    predictor_config.epoch_update = current_epochs.update;
    predictor_config.lr_update = current_lr.update;

    return predictor_config;
}

ValueRangeConfig init_value_range_config(const std::string& data_source, float& minimal_threshold) {
    ValueRangeConfig config;
    const Config& cfg = Config::getInstance();
    
    // Get bounds
    std::string lower_key = data_source + ".bounds.lower";
    std::string upper_key = data_source + ".bounds.upper";
    std::string threshold_key = data_source + ".minimal_threshold";
    
    config.lower_bound = std::stof(cfg.get_config_map().at(lower_key));
    config.upper_bound = std::stof(cfg.get_config_map().at(upper_key));
    minimal_threshold = std::stof(cfg.get_config_map().at(threshold_key));
    
    return config;
}

const OptimizerConfig::Epochs& Config::get_current_epochs() const {
    return optimizer_config.type == "adam" ? 
           optimizer_config.adam.epochs : 
           optimizer_config.sgd.epochs;
}

const OptimizerConfig::LearningRates& Config::get_current_learning_rates() const {
    return optimizer_config.type == "adam" ? 
           optimizer_config.adam.learning_rates : 
           optimizer_config.sgd.learning_rates;
}
