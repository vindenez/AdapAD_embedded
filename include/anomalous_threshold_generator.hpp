#ifndef ANOMALOUS_THRESHOLD_GENERATOR_HPP
#define ANOMALOUS_THRESHOLD_GENERATOR_HPP

#include "lstm_predictor.hpp"
#include <vector>
#include <tuple>

class AnomalousThresholdGenerator {
public:
    AnomalousThresholdGenerator(int lookback_len, int prediction_len, float lower_bound, float upper_bound);
    AnomalousThresholdGenerator(
        const std::vector<std::vector<float>>& weight_ih_input,
        const std::vector<std::vector<float>>& weight_hh_input,
        const std::vector<float>& bias_ih_input,
        const std::vector<float>& bias_hh_input,
        const std::vector<std::vector<float>>& weight_ih_forget,
        const std::vector<std::vector<float>>& weight_hh_forget,
        const std::vector<float>& bias_ih_forget,
        const std::vector<float>& bias_hh_forget,
        const std::vector<std::vector<float>>& weight_ih_output,
        const std::vector<std::vector<float>>& weight_hh_output,
        const std::vector<float>& bias_ih_output,
        const std::vector<float>& bias_hh_output,
        const std::vector<std::vector<float>>& weight_ih_cell,
        const std::vector<std::vector<float>>& weight_hh_cell,
        const std::vector<float>& bias_ih_cell,
        const std::vector<float>& bias_hh_cell);

    float generate(const std::vector<float>& prediction_errors, float minimal_threshold);
    void update(float learning_rate, const std::vector<float>& past_errors);
    
    // Add these two methods
    float generate_threshold(const std::vector<float>& new_input);
    std::vector<float> generate_thresholds(const std::vector<std::vector<float>>& input_sequence);

private:
    int lookback_len;
    int prediction_len;
    float lower_bound;
    float upper_bound;
    LSTMPredictor generator;
    std::vector<float> h;
    std::vector<float> c;
    std::vector<float> output;

    // Adam optimizer parameters
    std::vector<std::vector<float>> m_w_ih, m_w_hh;
    std::vector<float> m_b_ih, m_b_hh;
    std::vector<std::vector<float>> v_w_ih, v_w_hh;
    std::vector<float> v_b_ih, v_b_hh;
    int t; // time step for Adam

    // Adam hyperparameters
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    std::tuple<std::vector<float>, std::vector<float>, 
               std::vector<std::vector<float>>, std::vector<std::vector<float>>,
               std::vector<float>, std::vector<float>>
    backward_step(const std::vector<float>& input,
                  const std::vector<float>& h_prev,
                  const std::vector<float>& c_prev,
                  const std::vector<float>& doutput);

    void update_parameters(const std::vector<std::vector<float>>& dw_ih,
                           const std::vector<std::vector<float>>& dw_hh,
                           const std::vector<float>& db_ih,
                           const std::vector<float>& db_hh,
                           float learning_rate);

    void update_parameters_adam(const std::vector<std::vector<float>>& dw_ih,
                                const std::vector<std::vector<float>>& dw_hh,
                                const std::vector<float>& db_ih,
                                const std::vector<float>& db_hh,
                                float learning_rate);

    void init_adam_parameters();
};

#endif // ANOMALOUS_THRESHOLD_GENERATOR_HPP
