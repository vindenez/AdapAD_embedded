#include "optimizer.hpp"
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>


// Helper function for pow with floats
inline float pow_float(float base, float exp) {
    return std::pow(base, exp);
}

// Adam Implementation
Adam::Adam(float beta1, float beta2, float epsilon)
    : beta1(beta1), beta2(beta2), epsilon(epsilon), 
      timestep(0), is_initialized(false),
      num_layers(0), hidden_size(0), input_size(0), num_classes(0) {}

void Adam::reset() {
    // Clear all momentum buffers first
    m_fc_weight.clear();
    v_fc_weight.clear();
    m_fc_bias.clear();
    v_fc_bias.clear();
    
    m_weight_ih.clear();
    v_weight_ih.clear();
    m_weight_hh.clear();
    v_weight_hh.clear();
    m_bias_ih.clear();
    v_bias_ih.clear();
    m_bias_hh.clear();
    v_bias_hh.clear();
    
    // Reset state
    timestep = 0;
    is_initialized = false;
}

void Adam::init(int num_layers, int hidden_size, int input_size, int num_classes) {
    // Clear existing state first
    reset();
    
    float k = 0.0f;
    
    try {
        // Store dimensions
        this->num_layers = num_layers;
        this->hidden_size = hidden_size;
        this->input_size = input_size;
        this->num_classes = num_classes;
        
        // Initialize FC layer states
        m_fc_weight.resize(num_classes, std::vector<float>(hidden_size, k));
        v_fc_weight.resize(num_classes, std::vector<float>(hidden_size, k));
        m_fc_bias.resize(num_classes, k);
        v_fc_bias.resize(num_classes, k);
        
        // Initialize LSTM layer states
        m_weight_ih.resize(num_layers);
        v_weight_ih.resize(num_layers);
        m_weight_hh.resize(num_layers);
        v_weight_hh.resize(num_layers);
        m_bias_ih.resize(num_layers);
        v_bias_ih.resize(num_layers);
        m_bias_hh.resize(num_layers);
        v_bias_hh.resize(num_layers);
        
        // Initialize each layer's states
        for (int layer = 0; layer < num_layers; ++layer) {
            int input_size_layer = (layer == 0) ? input_size : hidden_size;
            
            m_weight_ih[layer].resize(4 * hidden_size, std::vector<float>(input_size_layer, k));
            v_weight_ih[layer].resize(4 * hidden_size, std::vector<float>(input_size_layer, k));
            m_weight_hh[layer].resize(4 * hidden_size, std::vector<float>(hidden_size, k));
            v_weight_hh[layer].resize(4 * hidden_size, std::vector<float>(hidden_size, k));
            m_bias_ih[layer].resize(4 * hidden_size, k);
            v_bias_ih[layer].resize(4 * hidden_size, k);
            m_bias_hh[layer].resize(4 * hidden_size, k);
            v_bias_hh[layer].resize(4 * hidden_size, k);
        }
        
        is_initialized = true;
        
    } catch (const std::exception& e) {
        // Clean up on failure
        reset();
        throw std::runtime_error("Failed to initialize Adam optimizer: " + std::string(e.what()));
    }
}

void Adam::update(std::vector<std::vector<float>>& weights,
                 std::vector<std::vector<float>>& grads,
                 float learning_rate) {
    if (!is_initialized) {
        throw std::runtime_error("Adam optimizer not initialized");
    }
    
    // Check for empty vectors
    if (weights.empty() || grads.empty()) {
        throw std::runtime_error("Empty vectors in Adam update");
    }

    // Check for empty inner vectors
    if (weights[0].empty() || grads[0].empty()) {
        throw std::runtime_error("Empty inner vectors in Adam update");
    }

    // Get the correct momentum buffers based on dimensions
    std::vector<std::vector<float>>* m_ptr = nullptr;
    std::vector<std::vector<float>>* v_ptr = nullptr;

    if (weights.size() == num_classes && weights[0].size() == hidden_size) {
        m_ptr = &m_fc_weight;
        v_ptr = &v_fc_weight;
    } else {
        for (int layer = 0; layer < num_layers; ++layer) {
            int input_size_layer = (layer == 0) ? input_size : hidden_size;
            if (weights.size() == 4 * hidden_size && weights[0].size() == input_size_layer) {
                m_ptr = &m_weight_ih[layer];
                v_ptr = &v_weight_ih[layer];
                break;
            }
            if (weights.size() == 4 * hidden_size && weights[0].size() == hidden_size) {
                m_ptr = &m_weight_hh[layer];
                v_ptr = &v_weight_hh[layer];
                break;
            }
        }
    }

    if (!m_ptr || !v_ptr) {
        throw std::runtime_error("Could not find matching momentum buffers");
    }

    // Dimension checks
    if (weights.size() != grads.size() || 
        weights[0].size() != grads[0].size() ||
        weights.size() != m_ptr->size() ||
        weights[0].size() != (*m_ptr)[0].size() ||
        weights.size() != v_ptr->size() ||
        weights[0].size() != (*v_ptr)[0].size()) {
        throw std::runtime_error("Dimension mismatch in Adam update");
    }

    timestep++;
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            // Update biased moments
            (*m_ptr)[i][j] = beta1 * (*m_ptr)[i][j] + (1.0f - beta1) * grads[i][j];
            (*v_ptr)[i][j] = beta2 * (*v_ptr)[i][j] + (1.0f - beta2) * grads[i][j] * grads[i][j];
            
            // Compute bias-corrected moments
            float m_hat = (*m_ptr)[i][j] / (1.0f - pow_float(beta1, static_cast<float>(timestep)));
            float v_hat = (*v_ptr)[i][j] / (1.0f - pow_float(beta2, static_cast<float>(timestep)));
            
            // Update weights
            weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

void Adam::update(std::vector<float>& biases,
                 std::vector<float>& grads,
                 float learning_rate) {
    if (!is_initialized) {
        throw std::runtime_error("Adam optimizer not initialized");
    }
    
    // Get the correct momentum buffers based on dimensions
    std::vector<float>* m_ptr = nullptr;
    std::vector<float>* v_ptr = nullptr;

    if (biases.size() == num_classes) {
        m_ptr = &m_fc_bias;
        v_ptr = &v_fc_bias;
    } else {
        for (int layer = 0; layer < num_layers; ++layer) {
            if (biases.size() == 4 * hidden_size) {
                if (m_bias_ih[layer].size() == biases.size()) {
                    m_ptr = &m_bias_ih[layer];
                    v_ptr = &v_bias_ih[layer];
                    break;
                }
                if (m_bias_hh[layer].size() == biases.size()) {
                    m_ptr = &m_bias_hh[layer];
                    v_ptr = &v_bias_hh[layer];
                    break;
                }
            }
        }
    }

    if (!m_ptr || !v_ptr) {
        throw std::runtime_error("Could not find matching bias momentum buffers");
    }

    if (biases.size() != grads.size() || 
        biases.size() != m_ptr->size() || 
        biases.size() != v_ptr->size()) {
        throw std::runtime_error("Dimension mismatch in Adam bias update");
    }

    timestep++;
    
    for (size_t i = 0; i < biases.size(); ++i) {
        // Update biased moments
        (*m_ptr)[i] = beta1 * (*m_ptr)[i] + (1.0f - beta1) * grads[i];
        (*v_ptr)[i] = beta2 * (*v_ptr)[i] + (1.0f - beta2) * grads[i] * grads[i];
        
        // Compute bias-corrected moments
        float m_hat = (*m_ptr)[i] / (1.0f - pow_float(beta1, static_cast<float>(timestep)));
        float v_hat = (*v_ptr)[i] / (1.0f - pow_float(beta2, static_cast<float>(timestep)));
        
        // Update biases
        biases[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

bool Adam::initialized() const {
    return is_initialized;
}

// SGD Implementation
SGD::SGD(float learning_rate, float momentum, float weight_decay) 
    : learning_rate(learning_rate), beta(momentum), weight_decay(weight_decay), is_initialized(false) {}

void SGD::init(int num_layers, int hidden_size, int input_size, int num_classes) {
    float k = 0.0f;
    
    try {
        // Initialize FC layer momentum
        v_fc_weight.resize(num_classes, std::vector<float>(hidden_size, k));
        v_fc_bias.resize(num_classes, k);
        
        // Initialize LSTM layer momentum buffers directly
        std::vector<std::vector<std::vector<float>>> new_v_weight_ih(num_layers);
        std::vector<std::vector<std::vector<float>>> new_v_weight_hh(num_layers);
        std::vector<std::vector<float>> new_v_bias_ih(num_layers);
        std::vector<std::vector<float>> new_v_bias_hh(num_layers);
        
        // Initialize each layer's momentum
        for (int layer = 0; layer < num_layers; ++layer) {
            int input_size_layer = (layer == 0) ? input_size : hidden_size;
            
            new_v_weight_ih[layer] = std::vector<std::vector<float>>(
                4 * hidden_size, std::vector<float>(input_size_layer, k));
            new_v_weight_hh[layer] = std::vector<std::vector<float>>(
                4 * hidden_size, std::vector<float>(hidden_size, k));
            new_v_bias_ih[layer] = std::vector<float>(4 * hidden_size, k);
            new_v_bias_hh[layer] = std::vector<float>(4 * hidden_size, k);
        }

        // Assign new vectors to member variables
        v_weight_ih.swap(new_v_weight_ih);
        v_weight_hh.swap(new_v_weight_hh);
        v_bias_ih.swap(new_v_bias_ih);
        v_bias_hh.swap(new_v_bias_hh);

        is_initialized = true;
        
    } catch (const std::exception& e) {
        throw;
    }
}

void SGD::update(std::vector<std::vector<float>>& weights,
                std::vector<std::vector<float>>& grads,
                float lr) {
    if (!is_initialized) {
        throw std::runtime_error("SGD optimizer not initialized");
    }
    
    // Check for empty vectors
    if (weights.empty() || grads.empty()) {
        throw std::runtime_error("Empty vectors in SGD update");
    }

    // Check for empty inner vectors
    if (weights[0].empty() || grads[0].empty()) {
        throw std::runtime_error("Empty inner vectors in SGD update");
    }

    // Dimension checks
    if (weights.size() != grads.size() || weights[0].size() != grads[0].size()) {
        throw std::runtime_error("Dimension mismatch in SGD update");
    }

    float actual_lr = lr > 0.0f ? lr : learning_rate;
    
    for (size_t i = 0; i < weights.size(); ++i) {
        float* w = weights[i].data();
        float* g = grads[i].data();
        float* v = v_fc_weight[i].data();
        const size_t len = weights[i].size();
        
        for (size_t j = 0; j < len; ++j) {
            // Add weight decay
            g[j] += weight_decay * w[j];
            // Update momentum
            v[j] = beta * v[j] + g[j];
            // Update weights
            w[j] -= actual_lr * v[j];
        }
    }
}

void SGD::update(std::vector<float>& weights,
                std::vector<float>& grads,
                float lr) {
    if (!is_initialized) {
        throw std::runtime_error("SGD optimizer not initialized");
    }
    
    // Dimension check
    if (weights.size() != grads.size()) {
        throw std::runtime_error("Dimension mismatch in SGD bias update");
    }

    float actual_lr = lr > 0.0f ? lr : learning_rate;
    
    float* w = weights.data();
    float* g = grads.data();
    float* v = v_fc_bias.data();
    const size_t len = weights.size();
    
    for (size_t i = 0; i < len; ++i) {
        // Update momentum
        v[i] = beta * v[i] + g[i];
        // Update biases (no weight decay for biases)
        w[i] -= actual_lr * v[i];
    }
}

void SGD::reset() {
    is_initialized = false;
    
    v_fc_weight.clear();
    v_fc_bias.clear();
    v_weight_ih.clear();
    v_weight_hh.clear();
    v_bias_ih.clear();
    v_bias_hh.clear();
}

bool SGD::initialized() const {
    return is_initialized;
}