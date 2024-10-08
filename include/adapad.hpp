#ifndef ADAPAD_HPP
#define ADAPAD_HPP

#include "normal_data_predictor.hpp"
#include "anomalous_threshold_generator.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

class AdapAD {
public:
    AdapAD(const PredictorConfig& predictor_config, const ValueRangeConfig& value_range_config, float minimal_threshold);

    void set_training_data(const std::vector<float>& data);
    void train(); // Added train method declaration
    bool is_anomalous(float observed_val);
    void clean();
    void log_results();

private:
    NormalDataPredictor data_predictor;
    AnomalousThresholdGenerator generator;
    float minimal_threshold;
    PredictorConfig predictor_config;

    std::vector<float> observed_vals;
    std::vector<float> predicted_vals;
    std::vector<float> thresholds;
    std::vector<float> predictive_errors; // Added predictive errors vector

    std::ofstream f_log;

    std::unordered_map<std::string, std::vector<std::vector<float>>> load_weights();
    std::unordered_map<std::string, std::vector<float>> load_biases();
};

#endif // ADAPAD_HPP
