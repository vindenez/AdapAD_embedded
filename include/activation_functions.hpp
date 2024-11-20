#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <vector>
#include <cmath>

float sigmoid_func(float x);
float tanh_func(float x);
std::vector<float> sigmoid_vector(const std::vector<float>& x);
std::vector<float> tanh_vector(const std::vector<float>& x);
std::vector<float> sigmoid_derivative(const std::vector<float>& x);
std::vector<float> tanh_derivative(const std::vector<float>& x);

#endif // ACTIVATION_FUNCTIONS_HPP
