#pragma once
#include "lstm_predictor.hpp"
#include <arm_neon.h>

class LSTMPredictorQuantized : public LSTMPredictor {
public:
    struct QuantizedLSTMLayer {
        // Quantized weights (8-bit integers)
        std::vector<std::vector<int8_t>> weight_ih_q;
        std::vector<std::vector<int8_t>> weight_hh_q;
        std::vector<int8_t> bias_ih_q;
        std::vector<int8_t> bias_hh_q;
        
        // Scale factors for dequantization
        float weight_ih_scale;
        float weight_hh_scale;
        float bias_ih_scale;
        float bias_hh_scale;
        
        // Zero points
        int8_t weight_ih_zero_point = 0;
        int8_t weight_hh_zero_point = 0;
    };

    LSTMPredictorQuantized(int num_classes, int input_size, int hidden_size, 
                          int num_layers, int lookback_len, bool batch_first = true);
    
    // Override key methods to use quantized arithmetic
    std::vector<float> lstm_cell_forward(
        const std::vector<float>& input,
        std::vector<float>& h_state,
        std::vector<float>& c_state,
        const LSTMLayer& layer) override;
    
    // Quantization method to convert float model to quantized
    void quantize_model();
    
private:
    std::vector<QuantizedLSTMLayer> quantized_layers;
    
    // Quantization utilities
    void quantize_tensor(const std::vector<std::vector<float>>& float_tensor,
                        std::vector<std::vector<int8_t>>& int8_tensor,
                        float& scale, int8_t& zero_point);
    
    void quantize_tensor(const std::vector<float>& float_tensor,
                        std::vector<int8_t>& int8_tensor,
                        float& scale, int8_t& zero_point);
    
    // Dequantization utilities
    void dequantize_tensor(const std::vector<std::vector<int8_t>>& int8_tensor,
                          std::vector<std::vector<float>>& float_tensor,
                          float scale, int8_t zero_point);
    
    // NEON-optimized quantized operations
    void quantized_matrix_vector_multiply_neon(
        const std::vector<std::vector<int8_t>>& matrix_q,
        const std::vector<float>& vector,
        std::vector<float>& result,
        float matrix_scale,
        int8_t matrix_zero_point,
        float input_scale);
    
    // Quantize inputs on the fly
    std::vector<int8_t> quantize_input(const std::vector<float>& input,
                                      float& scale, int8_t& zero_point);
};