#ifndef ADAPAD_HPP
#define ADAPAD_HPP

#include "normal_data_predictor.hpp"
#include "anomalous_threshold_generator.hpp"
#include "lstm_predictor.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

class AdapAD {
public:
    AdapAD(const PredictorConfig& predictor_config, 
           const ValueRangeConfig& value_range_config, 
           float minimal_threshold,
           const LSTMPredictor& lstm_predictor);

    void set_training_data(const std::vector<float>& data);
    bool is_anomalous(float observed_val);
    void clean();
    void log_results();

private:
    NormalDataPredictor data_predictor;
    AnomalousThresholdGenerator generator;
    float minimal_threshold;
    PredictorConfig predictor_config;
    ValueRangeConfig value_range_config;
    std::vector<float> observed_vals;
    std::vector<float> predicted_vals;
    std::vector<float> thresholds;
    std::vector<float> predictive_errors;
    std::ofstream f_log;

    NormalDataPredictor create_normal_data_predictor(const LSTMPredictor& lstm_predictor);
};

#endif // ADAPAD_HPP
