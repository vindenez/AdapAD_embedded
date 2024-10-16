#ifndef ADAPAD_HPP
#define ADAPAD_HPP

#include "normal_data_predictor.hpp"
#include "anomalous_threshold_generator.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <fstream>

class AdapAD {
public:
    AdapAD(const PredictorConfig& predictor_config,
           const ValueRangeConfig& value_range_config,
           float minimal_threshold,
           NormalDataPredictor& data_predictor,
           const std::vector<float>& training_data);

    void set_training_data(const std::vector<float>& data);
    void clean();
    void log_results();
    bool is_anomalous(float observed_val, bool actual_anomaly, bool log_results);

    std::vector<float> prepare_data_for_prediction();
    void log_result(bool is_anomalous, float normalized_val, float predicted_val, float threshold, bool actual_anomaly);

    void warmup_generator(const std::vector<float>& normalized_data);


    // Public methods for data normalization
    float normalize_data(float val) const;
    float reverse_normalized_data(float val) const;
    std::string get_log_filename() const;

    void open_log_file();

private:
    NormalDataPredictor& data_predictor;
    AnomalousThresholdGenerator generator;
    float minimal_threshold;
    PredictorConfig predictor_config;
    ValueRangeConfig value_range_config;
    std::vector<float> observed_vals;
    std::vector<float> predicted_vals;
    std::vector<float> thresholds;
    std::vector<float> predictive_errors;
    std::ofstream f_log;

    // Private methods for internal processing
    bool is_inside_range(float val) const;
    
};

#endif // ADAPAD_HPP
