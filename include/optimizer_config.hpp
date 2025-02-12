#pragma once

#include <string>

struct OptimizerConfig {
    static const std::string DEFAULT_TYPE;
    
    struct Epochs {
        int train;
        int update;
        int update_generator;
    };

    struct LearningRates {
        float train;
        float update;
        float update_generator;
    };

    struct AdamConfig {
        Epochs epochs;
        LearningRates learning_rates;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
    };

    struct SGDConfig {
        Epochs epochs;
        LearningRates learning_rates;
    };

    std::string type;
    AdamConfig adam;
    SGDConfig sgd;
};