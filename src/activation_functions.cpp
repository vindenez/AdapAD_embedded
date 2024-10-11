#include "activation_functions.hpp"
#include <cmath>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float tanh_func(float x) {
    return std::tanh(x);
}

float d_tanh_func(float x) {
    float tanh_x = std::tanh(x);
    return 1.0f - tanh_x * tanh_x;
}
