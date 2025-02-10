#pragma once
#include <vector>
#include <memory>

struct LSTMOptimizerState {
    std::vector<std::vector<std::vector<float>>> m_weight_ih;
    std::vector<std::vector<std::vector<float>>> m_weight_hh;
    std::vector<std::vector<float>> m_bias_ih;
    std::vector<std::vector<float>> m_bias_hh;
    std::vector<std::vector<float>> m_fc_weight;
    std::vector<float> m_fc_bias;
    bool initialized = false;
    int timestep = 0;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void initialize_state(int num_layers, int input_size, int hidden_size, int num_classes) = 0;
    virtual bool is_state_initialized() const = 0;
    virtual void reset_state() = 0;
    
    virtual void update_weights(std::vector<std::vector<float>>& weights,
                              std::vector<std::vector<float>>& grads,
                              std::vector<std::vector<float>>& momentum) = 0;
                              
    virtual void update_biases(std::vector<float>& weights,
                             std::vector<float>& grads,
                             std::vector<float>& momentum) = 0;

    virtual void set_learning_rate(float lr) = 0;
    virtual void set_momentum(float momentum) = 0;
    virtual void set_weight_decay(float decay) = 0;

    const LSTMOptimizerState& get_state() const { return state; }
    LSTMOptimizerState& get_state() { return state; }
protected:
    LSTMOptimizerState state;
};

class SGD : public Optimizer {
private:
    float learning_rate;
    float beta;  // momentum coefficient
    float weight_decay;
    
public:
    SGD(float lr = 0.01f, float momentum = 0.9f, float decay = 0.0f);
    
    void initialize_state(int num_layers, int input_size, int hidden_size, int num_classes) override;
    bool is_state_initialized() const override;
    void reset_state() override;
    
    void update_weights(std::vector<std::vector<float>>& weights,
                       std::vector<std::vector<float>>& grads,
                       std::vector<std::vector<float>>& momentum) override;
                       
    void update_biases(std::vector<float>& weights,
                      std::vector<float>& grads,
                      std::vector<float>& momentum) override;

    void set_learning_rate(float lr) override { learning_rate = lr; }
    void set_momentum(float momentum) override { beta = momentum; }
    void set_weight_decay(float decay) override { weight_decay = decay; }
}; 