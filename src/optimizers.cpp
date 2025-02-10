#include "optimizers.hpp"

SGD::SGD(float lr, float momentum, float weight_decay) 
    : learning_rate(lr), beta(momentum), weight_decay(weight_decay) {}

void SGD::initialize_state(int num_layers, int input_size, int hidden_size, int num_classes) {
    // Clear existing state first
    state = LSTMOptimizerState();
    
    state.m_weight_ih.resize(num_layers);
    state.m_weight_hh.resize(num_layers);
    state.m_bias_ih.resize(num_layers);
    state.m_bias_hh.resize(num_layers);
    
    for (int layer = 0; layer < num_layers; ++layer) {
        int input_size_layer = (layer == 0) ? input_size : hidden_size;
        
        state.m_weight_ih[layer].resize(4 * hidden_size, std::vector<float>(input_size_layer, 0.0f));
        state.m_weight_hh[layer].resize(4 * hidden_size, std::vector<float>(hidden_size, 0.0f));
        state.m_bias_ih[layer].resize(4 * hidden_size, 0.0f);
        state.m_bias_hh[layer].resize(4 * hidden_size, 0.0f);
    }
    
    state.m_fc_weight.resize(num_classes, std::vector<float>(hidden_size, 0.0f));
    state.m_fc_bias.resize(num_classes, 0.0f);
    
    state.initialized = true;
    state.timestep = 0;
}

bool SGD::is_state_initialized() const {
    return state.initialized;
}

void SGD::reset_state() {
    state = LSTMOptimizerState();
}

void SGD::update_weights(std::vector<std::vector<float>>& weights,
                        std::vector<std::vector<float>>& grads,
                        std::vector<std::vector<float>>& momentum) {
    for (size_t i = 0; i < weights.size(); ++i) {
        float* w = weights[i].data();
        float* g = grads[i].data();
        float* m = momentum[i].data();
        const size_t len = weights[i].size();
        
        for (size_t j = 0; j < len; ++j) {
            // Add weight decay
            g[j] += weight_decay * w[j];
            // Update momentum
            m[j] = beta * m[j] + g[j];
            // Update weights
            w[j] -= learning_rate * m[j];
        }
    }
}

void SGD::update_biases(std::vector<float>& weights,
                       std::vector<float>& grads,
                       std::vector<float>& momentum) {
    float* w = weights.data();
    float* g = grads.data();
    float* m = momentum.data();
    const size_t len = weights.size();
    
    for (size_t i = 0; i < len; ++i) {
        // Update momentum
        m[i] = beta * m[i] + g[i];
        // Update biases (no weight decay for biases)
        w[i] -= learning_rate * m[i];
    }
}
