#pragma once
#include <vector>
#include <memory>
#include <string>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void update(std::vector<std::vector<float>>& weights, 
                       std::vector<std::vector<float>>& grads,
                       float learning_rate) = 0;
                       
    virtual void update(std::vector<float>& weights,
                       std::vector<float>& grads,
                       float learning_rate) = 0;
                       
    virtual void reset() = 0;
    virtual bool initialized() const = 0;
    virtual void init(int num_layers, int hidden_size, int input_size, int num_classes) = 0;
};

class Adam : public Optimizer {
public:
    Adam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    
    void update(std::vector<std::vector<float>>& weights, 
                std::vector<std::vector<float>>& grads,
                float learning_rate) override;
                
    void update(std::vector<float>& weights,
                std::vector<float>& grads,
                float learning_rate) override;
                
    void reset() override;
    bool initialized() const override;
    void init(int num_layers, int hidden_size, int input_size, int num_classes) override;

private:
    float beta1;
    float beta2;
    float epsilon;
    int timestep;
    bool is_initialized;
    
    // Layer dimensions
    int num_layers;
    int hidden_size;
    int input_size;
    int num_classes;
    
    // Momentum and velocity states for all parameters
    std::vector<std::vector<float>> m_fc_weight;
    std::vector<std::vector<float>> v_fc_weight;
    std::vector<float> m_fc_bias;
    std::vector<float> v_fc_bias;
    
    std::vector<std::vector<std::vector<float>>> m_weight_ih;
    std::vector<std::vector<std::vector<float>>> v_weight_ih;
    std::vector<std::vector<std::vector<float>>> m_weight_hh;
    std::vector<std::vector<std::vector<float>>> v_weight_hh;
    std::vector<std::vector<float>> m_bias_ih;
    std::vector<std::vector<float>> v_bias_ih;
    std::vector<std::vector<float>> m_bias_hh;
    std::vector<std::vector<float>> v_bias_hh;
};

class SGD : public Optimizer {
public:
    explicit SGD(float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
    
    void update(std::vector<std::vector<float>>& weights, 
                std::vector<std::vector<float>>& grads,
                float learning_rate) override;
                
    void update(std::vector<float>& weights,
                std::vector<float>& grads,
                float learning_rate) override;
                
    void reset() override;
    bool initialized() const override;
    void init(int num_layers, int hidden_size, int input_size, int num_classes) override;

private:
    float learning_rate;
    float beta;  // momentum factor
    float weight_decay;
    bool is_initialized;
    
    // Momentum buffers
    std::vector<std::vector<std::vector<float>>> v_weight_ih;
    std::vector<std::vector<std::vector<float>>> v_weight_hh;
    std::vector<std::vector<float>> v_bias_ih;
    std::vector<std::vector<float>> v_bias_hh;
    std::vector<std::vector<float>> v_fc_weight;
    std::vector<float> v_fc_bias;
};
