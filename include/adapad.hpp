#ifndef ADAPAD_HPP
#define ADAPAD_HPP

#include "normal_data_predictor.hpp"
#include "anomalous_threshold_generator.hpp"
#include "config.hpp"

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>

class AdapAD {
public:
    AdapAD(const PredictorConfig& predictor_config, 
           const ValueRangeConfig& value_range_config,
           float minimal_threshold,
           const std::string& parameter_name);
    
    void set_training_data(const std::vector<float>& data);
    void train();
    bool is_anomalous(float observed_val);
    void clean();

    std::string get_log_filename() const { return f_name; }

    // Add model state methods
    void save_models(const std::string& model_file);
    void load_models(const std::string& model_file);

private:
    // Configuration
    ValueRangeConfig value_range_config;
    PredictorConfig predictor_config;
    float minimal_threshold;
    
    // Learning components
    std::unique_ptr<NormalDataPredictor> data_predictor;
    std::unique_ptr<AnomalousThresholdGenerator> generator;
    
    // Data storage
    std::vector<float> observed_vals;
    std::vector<float> predicted_vals;
    std::vector<float> predictive_errors;
    std::vector<float> thresholds;
    std::vector<size_t> anomalies;
    
    // Logging
    std::ofstream f_log;
    std::string f_name;
    
    // Helper methods
    void learn_error_pattern(const std::vector<std::vector<std::vector<float>>>& trainX,
                           const std::vector<float>& trainY);
    void update_generator(const std::vector<float>& past_errors, float recent_error);
    std::vector<std::vector<std::vector<float>>> prepare_data_for_prediction(size_t supposed_anomalous_pos);
    void logging(bool is_anomalous_ret);
    float normalize_data(float val);
    float reverse_normalized_data(float val);
    bool is_inside_range(float val);
    bool is_default_normal();
    float simplify_error(const std::vector<float>& errors, float N_sigma = 0);

    const Config& config;  // Reference to config instance
    int update_count;  // Track number of updates
    std::string get_state_filename() const;
    void clean_old_saves(size_t keep_count);
};
#endif // ADAPAD_HPP
