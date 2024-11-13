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
           float minimal_threshold);

    ~AdapAD();

    void set_training_data(const std::vector<float>& data);
    bool is_anomalous(float val, bool actual_anomaly);
    void train(float measured_value);
    void clean();

    // Public methods for data normalization
    float normalize_data(float val) const;
    float reverse_normalized_data(float val) const;

    std::string get_log_filename() const {
        return f_name;
    }

private:
    NormalDataPredictor data_predictor;
    AnomalousThresholdGenerator generator;
    PredictorConfig predictor_config;
    ValueRangeDb sensor_range;
    float minimal_threshold;

    std::vector<float> observed_vals;
    std::vector<float> predicted_vals;
    std::vector<float> thresholds;
    std::vector<float> predictive_errors;
    std::vector<size_t> anomalies;

    std::string f_name;
    std::ofstream f_log;

    // Private methods
    bool is_inside_range(float val) const;
    std::vector<float> prepare_data_for_prediction();
    bool is_default_normal() const;
    void update_generator(const std::vector<float>& past_observations, float observed_val);
    void log_result(bool is_anomalous, float normalized_val, float predicted_val, float threshold, bool actual_anomaly);
    void open_log_file();
    std::vector<float> calc_error(const std::vector<float>& ground_truth, const std::vector<float>& predict);
    void maintain_memory();
    float normalize_threshold(float threshold) const;

};

#endif // ADAPAD_HPP
