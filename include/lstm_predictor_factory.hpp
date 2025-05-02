#pragma once
#include "lstm_predictor.hpp"
#include "lstm_predictor_neon.hpp"
#include <memory>
#include <iostream>
#include "config.hpp"

// Check for ARM architecture with NEON support
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define HAS_NEON_SUPPORT 1
#else
    #define HAS_NEON_SUPPORT 0
#endif

class LSTMPredictorFactory {
public:
    static std::unique_ptr<LSTMPredictor> create_predictor(
        int num_classes,
        int input_size,
        int hidden_size,
        int num_layers,
        int lookback_len,
        bool batch_first = true) {
        
        // Get global config setting
        const Config& config = Config::getInstance();
        
        bool use_neon = (config.use_neon && HAS_NEON_SUPPORT);
        
        // Output warning if NEON is available but not being used
        if (HAS_NEON_SUPPORT && !use_neon) {
            std::cout << "WARNING: NEON IS AVAILABLE BUT NOT IN USE. Set config.use_neon=true "
                      << "to enable NEON optimizations." << std::endl;
        }
        
        if (use_neon) {
            std::cout << "Creating NEON-optimized LSTM predictor" << std::endl;
            return std::unique_ptr<LSTMPredictor>(new LSTMPredictorNEON(
                num_classes,
                input_size,
                hidden_size,
                num_layers,
                lookback_len,
                batch_first
            ));
        } else {
            std::cout << "Creating LSTM predictor" << std::endl;
            return std::unique_ptr<LSTMPredictor>(new LSTMPredictor(
                num_classes,
                input_size,
                hidden_size,
                num_layers,
                lookback_len,
                batch_first
            ));
        }
    }
};