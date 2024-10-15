#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <vector>

float sigmoid(float x);
float tanh_func(float x);
float d_tanh_func(float x);
std::vector<float> tanh_vector(const std::vector<float>& vec);
std::vector<float> sigmoid_vector(const std::vector<float>& vec);

#endif // ACTIVATION_FUNCTIONS_HPP
