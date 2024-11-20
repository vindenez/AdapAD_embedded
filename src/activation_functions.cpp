#include "activation_functions.hpp"
#include <cmath>
#include <algorithm>

float sigmoid_func(float x) {
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
    std::transform(vec.begin(), vec.end(), result.begin(), [](float x) { return sigmoid_func(x); });
    return result;
}

std::vector<float> tanh_vector(const std::vector<float>& vec) {
    std::vector<float> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(), [](float x) { return tanh_func(x); });
    return result;
}

std::vector<float> sigmoid_derivative(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float sig = sigmoid_func(x[i]);
        result[i] = sig * (1.0f - sig);
    }
    return result;
}

std::vector<float> tanh_derivative(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float th = tanh_func(x[i]);
        result[i] = 1.0f - th * th;
    }
    return result;
}

