#include "activation_functions.hpp"
#include <cmath>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float tanh_func(float x) {
    return std::tanh(x);
}
