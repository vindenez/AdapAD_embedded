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

std::vector<float> sigmoid_vector(const std::vector<float>& vec) {
    std::vector<float> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(), [](float x) { return sigmoid(x); });
    return result;
}

std::vector<float> tanh_vector(const std::vector<float>& vec) {
    std::vector<float> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(), [](float x) { return tanh_func(x); });
    return result;
}

