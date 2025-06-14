#ifndef LSTM_PREDICTOR_FACTORY_HPP
#define LSTM_PREDICTOR_FACTORY_HPP

#include "config.hpp"
#include "lstm_predictor.hpp"
#include <iostream>
#include <memory>

// Check for ARM architecture with NEON support
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define HAS_NEON_SUPPORT 1
#include "lstm_predictor_neon.hpp"
#else
#define HAS_NEON_SUPPORT 0
#endif

class LSTMPredictorFactory {
  public:
    static std::unique_ptr<LSTMPredictor> create_predictor(int num_classes, int input_size,
                                                           int hidden_size, int num_layers,
                                                           int lookback_len,
                                                           bool batch_first = true,
                                                           int random_seed_param = -1) {

        // Get global config setting
        const Config &config = Config::getInstance();

        int random_seed = (random_seed_param == -1) ? config.random_seed : random_seed_param;
        bool use_neon = (config.use_neon && HAS_NEON_SUPPORT);

        // Output warning if NEON is available but not being used
        if (HAS_NEON_SUPPORT && !use_neon) {
            std::cout << "WARNING: NEON IS AVAILABLE BUT NOT IN USE. Set "
                         "config.use_neon=true "
                      << "to enable NEON optimizations." << std::endl;
        }

        if (use_neon) {
            std::cout << "Creating NEON-optimized LSTM predictor" << std::endl;
            return std::unique_ptr<LSTMPredictor>(new LSTMPredictorNEON(
                num_classes, input_size, hidden_size, num_layers, lookback_len, batch_first, random_seed));
        } else {
            std::cout << "Creating LSTM predictor" << std::endl;
            return std::unique_ptr<LSTMPredictor>(new LSTMPredictor(
                num_classes, input_size, hidden_size, num_layers, lookback_len, batch_first, random_seed));
        }
    }
};

#endif // LSTM_PREDICTOR_FACTORY_HPP