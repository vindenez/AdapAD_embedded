#pragma once
#include "lstm_predictor.hpp"
#include "lstm_predictor_32b_neon.hpp"
#include "lstm_predictor_16b_neon.hpp"
#include <memory>
#include <iostream>
#include <variant>
#include "config.hpp"
#include <arm_neon.h>


// Check for ARM architecture with NEON support
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define HAS_NEON_SUPPORT 1
#else
    #define HAS_NEON_SUPPORT 0
#endif

// An interface class for both 32-bit and 16-bit instances of LSTMPredictor
class ILSTMPredictor {
public:
    virtual ~ILSTMPredictor() = default;

    // Forward/backward methods
    virtual LSTMPredictor::LSTMOutput forward(const std::vector<std::vector<std::vector<float>>>& x) = 0;
    virtual std::vector<float> get_final_prediction(const LSTMPredictor::LSTMOutput& output) = 0;
    virtual void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                           const std::vector<float>& target,
                           const LSTMPredictor::LSTMOutput& lstm_output,
                           float learning_rate) = 0;
    
    // Training methods
    virtual void eval() = 0;
    virtual void train() = 0;
    virtual void reset_states() = 0;
    virtual bool is_training() const = 0;
    virtual bool is_online_learning() const = 0;
    virtual void learn() = 0;
    
    // Model persistence methods
    virtual void save_weights(std::ofstream& file) = 0;
    virtual void save_biases(std::ofstream& file) = 0;
    virtual void load_weights(std::ifstream& file) = 0;
    virtual void load_biases(std::ifstream& file) = 0;
    virtual void save_layer_cache(std::ofstream& file) const = 0;
    virtual void load_layer_cache(std::ifstream& file) = 0;
    virtual void initialize_layer_cache() = 0;
    virtual bool is_layer_cache_initialized() const = 0;
    virtual void clear_update_state() = 0;
};

// Adapter for 32-bit predictors
class LSTMPredictor32Adapter : public ILSTMPredictor {
public:
    LSTMPredictor32Adapter(std::unique_ptr<LSTMPredictor> predictor) 
        : predictor_(std::move(predictor)) {}
    
    // Forward/backward methods for 32-bit
    LSTMPredictor::LSTMOutput forward(const std::vector<std::vector<std::vector<float>>>& x) {
        return predictor_->forward(x);
    }
    
    std::vector<float> get_final_prediction(const LSTMPredictor::LSTMOutput& output) {
        return predictor_->get_final_prediction(output);
    }
    
    void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                    const std::vector<float>& target,
                    const LSTMPredictor::LSTMOutput& lstm_output,
                    float learning_rate) {
        predictor_->train_step(x, target, lstm_output, learning_rate);
    }
    
    // Interface implementations
    void eval() override { predictor_->eval(); }
    void train() override { predictor_->train(); }
    void reset_states() override { predictor_->reset_states(); }
    bool is_training() const override { return predictor_->is_training(); }
    bool is_online_learning() const override { return predictor_->is_online_learning(); }
    void learn() override { predictor_->learn(); }
    
    void save_weights(std::ofstream& file) override { predictor_->save_weights(file); }
    void save_biases(std::ofstream& file) override { predictor_->save_biases(file); }
    void load_weights(std::ifstream& file) override { predictor_->load_weights(file); }
    void load_biases(std::ifstream& file) override { predictor_->load_biases(file); }
    void save_layer_cache(std::ofstream& file) const override { predictor_->save_layer_cache(file); }
    void load_layer_cache(std::ifstream& file) override { predictor_->load_layer_cache(file); }
    void initialize_layer_cache() override { predictor_->initialize_layer_cache(); }
    bool is_layer_cache_initialized() const override { return predictor_->is_layer_cache_initialized(); }
    void clear_update_state() override { predictor_->clear_update_state(); }
    
private:
    std::unique_ptr<LSTMPredictor> predictor_;
};

// Adapter for 16-bit predictors with automatic conversion
class LSTMPredictor16Adapter : public ILSTMPredictor {
public:
    LSTMPredictor16Adapter(std::unique_ptr<LSTMPredictor16bNEON> predictor) 
        : predictor_(std::move(predictor)) {}
    
    // Core methods with type conversion (32-bit interface -> 16-bit internal)
    LSTMPredictor::LSTMOutput forward(const std::vector<std::vector<std::vector<float>>>& x) override {
        auto x_16bit = convertToFloat16(x);
        
        // Store the actual 16-bit output for use in get_final_prediction()
        last_16bit_output_ = predictor_->forward(x_16bit);
        has_valid_16bit_output_ = true;
        
        // Convert and return as 32-bit output
        return convertToFloat32(last_16bit_output_);
    }
    
    std::vector<float> get_final_prediction(const LSTMPredictor::LSTMOutput& output) override {
        // Use the stored 16-bit output (more accurate than converting back from 32-bit)
        if (!has_valid_16bit_output_) {
            throw std::runtime_error("Must call forward() before get_final_prediction()");
        }
        auto pred_16bit = predictor_->get_final_prediction(last_16bit_output_);
        return convertToFloat32(pred_16bit);
    }
    
    void train_step(const std::vector<std::vector<std::vector<float>>>& x,
                    const std::vector<float>& target,
                    const LSTMPredictor::LSTMOutput& lstm_output,
                    float learning_rate) override {
        auto x_16bit = convertToFloat16(x);
        auto target_16bit = convertToFloat16(target);
        float16_t lr_16bit = static_cast<float16_t>(learning_rate);
        
        // We need to ensure we have a valid 16-bit output for training
        if (!has_valid_16bit_output_) {
            last_16bit_output_ = predictor_->forward(x_16bit);
            has_valid_16bit_output_ = true;
        }
        
        predictor_->train_step(x_16bit, target_16bit, last_16bit_output_, lr_16bit);
        has_valid_16bit_output_ = false;  // Invalidate after training
    }
    
    // State management - direct delegation
    void eval() override { predictor_->eval(); }
    void train() override { predictor_->train(); }
    void reset_states() override { 
        predictor_->reset_states();
        has_valid_16bit_output_ = false;
    }
    bool is_training() const override { return predictor_->is_training(); }
    bool is_online_learning() const override { return predictor_->is_online_learning(); }
    void learn() override { predictor_->learn(); }
    void clear_update_state() override { 
        predictor_->clear_update_state();
        has_valid_16bit_output_ = false;
    }
    
    // Layer cache
    void initialize_layer_cache() override { predictor_->initialize_layer_cache(); }
    bool is_layer_cache_initialized() const override { return predictor_->is_layer_cache_initialized(); }
    
    // Model persistence
    void save_weights(std::ofstream& file) override { predictor_->save_weights(file); }
    void load_weights(std::ifstream& file) override { predictor_->load_weights(file); }
    void save_biases(std::ofstream& file) override { predictor_->save_biases(file); }
    void load_biases(std::ifstream& file) override { predictor_->load_biases(file); }
    void save_layer_cache(std::ofstream& file) const override { predictor_->save_layer_cache(file); }
    void load_layer_cache(std::ifstream& file) override { predictor_->load_layer_cache(file); }
    
private:
    std::unique_ptr<LSTMPredictor16bNEON> predictor_;
    LSTMPredictor16bNEON::LSTMOutput16bit last_16bit_output_;
    bool has_valid_16bit_output_ = false;
    
    // Conversion utilities
    std::vector<std::vector<std::vector<float16_t>>> convertToFloat16(
        const std::vector<std::vector<std::vector<float>>>& input) {
        std::vector<std::vector<std::vector<float16_t>>> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i].resize(input[i].size());
            for (size_t j = 0; j < input[i].size(); ++j) {
                output[i][j].resize(input[i][j].size());
                for (size_t k = 0; k < input[i][j].size(); ++k) {
                    output[i][j][k] = static_cast<float16_t>(input[i][j][k]);
                }
            }
        }
        return output;
    }
    
    std::vector<float16_t> convertToFloat16(const std::vector<float>& input) {
        std::vector<float16_t> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = static_cast<float16_t>(input[i]);
        }
        return output;
    }
    
    std::vector<float> convertToFloat32(const std::vector<float16_t>& input) {
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = static_cast<float>(input[i]);
        }
        return output;
    }
    
    LSTMPredictor::LSTMOutput convertToFloat32(const LSTMPredictor16bNEON::LSTMOutput16bit& input) {
        LSTMPredictor::LSTMOutput output;
        
        // Convert sequence_output
        output.sequence_output.resize(input.sequence_output.size());
        for (size_t i = 0; i < input.sequence_output.size(); ++i) {
            output.sequence_output[i].resize(input.sequence_output[i].size());
            for (size_t j = 0; j < input.sequence_output[i].size(); ++j) {
                output.sequence_output[i][j].resize(input.sequence_output[i][j].size());
                for (size_t k = 0; k < input.sequence_output[i][j].size(); ++k) {
                    output.sequence_output[i][j][k] = static_cast<float>(input.sequence_output[i][j][k]);
                }
            }
        }
        
        // Convert final_hidden
        output.final_hidden.resize(input.final_hidden.size());
        for (size_t i = 0; i < input.final_hidden.size(); ++i) {
            output.final_hidden[i].resize(input.final_hidden[i].size());
            for (size_t j = 0; j < input.final_hidden[i].size(); ++j) {
                output.final_hidden[i][j] = static_cast<float>(input.final_hidden[i][j]);
            }
        }
        
        // Convert final_cell
        output.final_cell.resize(input.final_cell.size());
        for (size_t i = 0; i < input.final_cell.size(); ++i) {
            output.final_cell[i].resize(input.final_cell[i].size());
            for (size_t j = 0; j < input.final_cell[i].size(); ++j) {
                output.final_cell[i][j] = static_cast<float>(input.final_cell[i][j]);
            }
        }
        
        return output;
    }
};

class LSTMPredictorFactory {
public:
    static std::unique_ptr<ILSTMPredictor> create_predictor(
        int num_classes,
        int input_size,
        int hidden_size,
        int num_layers,
        int lookback_len,
        bool batch_first = true) {
        
        const Config& config = Config::getInstance();
        
        bool use_neon = (config.use_neon && HAS_NEON_SUPPORT);
        bool use_16bit = config.use_16bit;
        
        if (HAS_NEON_SUPPORT && !use_neon) {
            std::cout << "WARNING: NEON IS AVAILABLE BUT NOT IN USE." << std::endl;
        }
        
        if (use_neon && use_16bit) {
            std::cout << "Creating 16-bit NEON-optimized LSTM predictor" << std::endl;
            std::unique_ptr<LSTMPredictor16bNEON> predictor_16bit(
                new LSTMPredictor16bNEON(num_classes, input_size, hidden_size, 
                                         num_layers, lookback_len, batch_first));
            return std::unique_ptr<ILSTMPredictor>(
                new LSTMPredictor16Adapter(std::move(predictor_16bit)));
        } else if (use_neon && !use_16bit) {
            std::cout << "Creating 32-bit NEON-optimized LSTM predictor" << std::endl;
            std::unique_ptr<LSTMPredictor32bNEON> predictor_32bit(
                new LSTMPredictor32bNEON(num_classes, input_size, hidden_size, 
                                         num_layers, lookback_len, batch_first));
            return std::unique_ptr<ILSTMPredictor>(
                new LSTMPredictor32Adapter(std::move(predictor_32bit)));
        } else {
            std::cout << "Creating standard LSTM predictor" << std::endl;
            std::unique_ptr<LSTMPredictor> predictor_32bit(
                new LSTMPredictor(num_classes, input_size, hidden_size, 
                                  num_layers, lookback_len, batch_first));
            return std::unique_ptr<ILSTMPredictor>(
                new LSTMPredictor32Adapter(std::move(predictor_32bit)));
        }
    }
};