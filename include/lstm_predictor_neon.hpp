#ifndef LSTM_PREDICTOR_NEON_HPP
#define LSTM_PREDICTOR_NEON_HPP

#include "lstm_predictor.hpp"
#include <arm_neon.h>
#include <cmath>
#include <vector>

class LSTMPredictorNEON : public LSTMPredictor {
  public:
    // Constructor
    LSTMPredictorNEON(int num_classes, int input_size, int hidden_size, int num_layers,
                      int lookback_len, bool batch_first = true);

    // Override LSTM cell forward method to use NEON optimizations
    std::vector<float> lstm_cell_forward(const std::vector<float> &input,
                                         std::vector<float> &h_state, std::vector<float> &c_state,
                                         const LSTMLayer &layer) override;

    LSTMOutput forward(const std::vector<std::vector<std::vector<float>>> &x,
                       const std::vector<std::vector<float>> *initial_hidden = nullptr,
                       const std::vector<std::vector<float>> *initial_cell = nullptr) override;

    void train_step(const std::vector<std::vector<std::vector<float>>> &x,
                    const std::vector<float> &target, const LSTMOutput &lstm_output,
                    float learning_rate) override;

    std::vector<float> get_final_prediction(const LSTMOutput &lstm_output) override;

    std::vector<LSTMGradients>
    backward_lstm_layer(const std::vector<float> &grad_output,
                        const std::vector<std::vector<std::vector<LSTMCacheEntry>>> &cache,
                        float learning_rate) override;

    void backward_linear_layer(const std::vector<float> &grad_output,
                               const std::vector<float> &last_hidden,
                               std::vector<std::vector<float>> &weight_grad,
                               std::vector<float> &bias_grad,
                               std::vector<float> &input_grad) override;

    // Add virtual destructor
    virtual ~LSTMPredictorNEON() = default;

  private:
    // NEON-optimized activation functions
    float32x4_t sigmoid_neon(float32x4_t x);
    float32x4_t tanh_neon(float32x4_t x);

    std::vector<float> mse_loss_gradient(const std::vector<float> &output,
                                         const std::vector<float> &target) override;
    float mse_loss(const std::vector<float> &prediction, const std::vector<float> &target) override;

    // NEON-optimized gate operations
    void apply_gate_operations_neon(std::vector<float> &gates, std::vector<float> &h_state,
                                    std::vector<float> &c_state, LSTMCacheEntry *cache_entry,
                                    size_t hidden_size);

    // Override optimizer functions to use NEON
    void apply_sgd_update(std::vector<std::vector<float>> &weights,
                          std::vector<std::vector<float>> &grads, float learning_rate,
                          float momentum = 0.9f) override;

    void apply_sgd_update(std::vector<float> &biases, std::vector<float> &grads,
                          float learning_rate, float momentum = 0.9f) override;
};

#endif // LSTM_PREDICTOR_NEON_HPP
