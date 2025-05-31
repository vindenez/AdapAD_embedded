#include "lstm_predictor.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <cassert>
#include <iomanip>

class LSTMTestSuite {
private:
    static constexpr float EPSILON = 1e-6f;
    static constexpr float TOLERANCE = 1e-4f;
    int test_count = 0;
    int passed_tests = 0;

    void assert_true(bool condition, const std::string& message) {
        test_count++;
        if (condition) {
            passed_tests++;
            std::cout << "✓ PASS: " << message << std::endl;
        } else {
            std::cout << "✗ FAIL: " << message << std::endl;
        }
    }

    void assert_near(float actual, float expected, float tolerance, const std::string& message) {
        test_count++;
        if (std::abs(actual - expected) <= tolerance) {
            passed_tests++;
            std::cout << "✓ PASS: " << message << " (actual: " << actual << ", expected: " << expected << ")" << std::endl;
        } else {
            std::cout << "✗ FAIL: " << message << " (actual: " << actual << ", expected: " << expected << ", diff: " << std::abs(actual - expected) << ")" << std::endl;
        }
    }

    void assert_vector_near(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const std::string& message) {
        test_count++;
        bool all_close = true;
        if (actual.size() != expected.size()) {
            all_close = false;
        } else {
            for (size_t i = 0; i < actual.size(); ++i) {
                if (std::abs(actual[i] - expected[i]) > tolerance) {
                    all_close = false;
                    break;
                }
            }
        }
        
        if (all_close) {
            passed_tests++;
            std::cout << "✓ PASS: " << message << std::endl;
        } else {
            std::cout << "✗ FAIL: " << message << std::endl;
            std::cout << "  Actual  : [";
            for (size_t i = 0; i < std::min(actual.size(), size_t(5)); ++i) {
                std::cout << actual[i] << (i < std::min(actual.size(), size_t(5)) - 1 ? ", " : "");
            }
            if (actual.size() > 5) std::cout << "...";
            std::cout << "]" << std::endl;
            std::cout << "  Expected: [";
            for (size_t i = 0; i < std::min(expected.size(), size_t(5)); ++i) {
                std::cout << expected[i] << (i < std::min(expected.size(), size_t(5)) - 1 ? ", " : "");
            }
            if (expected.size() > 5) std::cout << "...";
            std::cout << "]" << std::endl;
        }
    }

    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>> 
    generate_test_data(int batch_size, int seq_length, int input_size, int num_classes) {
        std::vector<std::vector<std::vector<float>>> inputs(batch_size);
        std::vector<float> targets(num_classes, 0.0f);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (int b = 0; b < batch_size; ++b) {
            inputs[b].resize(seq_length);
            for (int t = 0; t < seq_length; ++t) {
                inputs[b][t].resize(input_size);
                for (int i = 0; i < input_size; ++i) {
                    float base_value = std::sin(2.0f * M_PI * t / seq_length + i);
                    inputs[b][t][i] = base_value + 0.1f * dist(gen);
                }
            }
        }
        
        for (int i = 0; i < std::min(num_classes, input_size); ++i) {
            targets[i] = std::tanh(inputs[0][seq_length-1][i]);
        }
        
        return {inputs, targets};
    }

    std::vector<std::vector<float>> copy_2d_weights(const std::vector<std::vector<float>>& weights) {
        std::vector<std::vector<float>> copy(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            copy[i] = weights[i];
        }
        return copy;
    }

    std::vector<float> copy_1d_weights(const std::vector<float>& weights) {
        return weights;
    }

    bool weights_changed(const std::vector<std::vector<float>>& before, const std::vector<std::vector<float>>& after) {
        if (before.size() != after.size()) return true;
        for (size_t i = 0; i < before.size(); ++i) {
            if (before[i].size() != after[i].size()) return true;
            for (size_t j = 0; j < before[i].size(); ++j) {
                if (std::abs(before[i][j] - after[i][j]) > EPSILON) return true;
            }
        }
        return false;
    }

    bool weights_changed(const std::vector<float>& before, const std::vector<float>& after) {
        if (before.size() != after.size()) return true;
        for (size_t i = 0; i < before.size(); ++i) {
            if (std::abs(before[i] - after[i]) > EPSILON) return true;
        }
        return false;
    }

public:
    void run_all_tests() {
        std::cout << "=== LSTM Predictor Test Suite ===" << std::endl;
        
        test_initialization();
        test_forward_pass();
        test_training_basic();
        test_parameter_updates();
        test_loss_computation();
        test_gradient_flow();
        test_model_persistence();
        test_state_management();
        test_error_handling();
        
        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Passed: " << passed_tests << "/" << test_count << std::endl;
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * passed_tests / test_count) << "%" << std::endl;
    }

    void test_initialization() {
        std::cout << "\n--- Testing Initialization ---" << std::endl;
        
        try {
            LSTMPredictor lstm(4, 3, 8, 2, 10, true, 42);
            assert_true(true, "LSTM initialization successful");
            
            assert_true(lstm.get_num_layers() == 2, "Number of layers set correctly");
            assert_true(lstm.is_training() == true, "Default training mode is true");
            
            auto weights = lstm.get_weights();
            assert_true(weights.size() == 2, "Correct number of LSTM layers initialized");
            
            if (weights.size() > 0) {
                assert_true(weights[0].weight_ih.size() == 4 * 8, "First layer weight_ih has correct dimensions (4*hidden_size)");
                assert_true(weights[0].weight_hh.size() == 4 * 8, "First layer weight_hh has correct dimensions (4*hidden_size)");
                assert_true(weights[0].bias_ih.size() == 4 * 8, "First layer bias_ih has correct dimensions");
                assert_true(weights[0].bias_hh.size() == 4 * 8, "First layer bias_hh has correct dimensions");
                
                if (weights[0].weight_ih.size() > 0) {
                    assert_true(weights[0].weight_ih[0].size() == 3, "Input-to-hidden weights have correct input dimension");
                }
                if (weights[0].weight_hh.size() > 0) {
                    assert_true(weights[0].weight_hh[0].size() == 8, "Hidden-to-hidden weights have correct hidden dimension");
                }
            }
            
            if (weights.size() > 1) {
                if (weights[1].weight_ih.size() > 0) {
                    assert_true(weights[1].weight_ih[0].size() == 8, "Second layer input size equals hidden size");
                }
            }
            
        } catch (const std::exception& e) {
            assert_true(false, "LSTM initialization failed: " + std::string(e.what()));
        }
    }

    void test_forward_pass() {
        std::cout << "\n--- Testing Forward Pass ---" << std::endl;
        
        LSTMPredictor lstm(3, 4, 6, 1, 5, true, 42);
        auto [inputs, targets] = generate_test_data(1, 5, 4, 3);
        
        try {
            lstm.eval();
            std::vector<float> output = lstm.forward(inputs);
            
            assert_true(output.size() == 3, "Output size matches num_classes");
            
            bool all_finite = true;
            for (float val : output) {
                if (!std::isfinite(val)) {
                    all_finite = false;
                    break;
                }
            }
            assert_true(all_finite, "All output values are finite");
            
            std::vector<float> output2 = lstm.forward(inputs);
            assert_vector_near(output, output2, EPSILON, "Forward pass is deterministic");
            
        } catch (const std::exception& e) {
            assert_true(false, "Forward pass failed: " + std::string(e.what()));
        }
    }

    void test_training_basic() {
        std::cout << "\n--- Testing Basic Training ---" << std::endl;
        
        LSTMPredictor lstm(2, 3, 4, 1, 6, true, 42);
        auto [inputs, targets] = generate_test_data(1, 6, 3, 2);
        
        try {
            lstm.train();
            
            float loss1 = lstm.train_step(inputs, targets, 0.01f);
            assert_true(std::isfinite(loss1), "First training step produces finite loss");
            assert_true(loss1 >= 0.0f, "Loss is non-negative");
            
            float loss2 = lstm.train_step(inputs, targets, 0.01f);
            assert_true(std::isfinite(loss2), "Second training step produces finite loss");
            
            for (int i = 0; i < 10; ++i) {
                lstm.train_step(inputs, targets, 0.01f);
            }
            
            float final_loss = lstm.train_step(inputs, targets, 0.01f);
            assert_true(final_loss < loss1, "Loss decreases after multiple training steps");
            
        } catch (const std::exception& e) {
            assert_true(false, "Training failed: " + std::string(e.what()));
        }
    }

    void test_parameter_updates() {
        std::cout << "\n--- Testing Parameter Updates ---" << std::endl;
        
        LSTMPredictor lstm(2, 3, 4, 1, 5, true, 42);
        auto [inputs, targets] = generate_test_data(1, 5, 3, 2);
        
        try {
            lstm.train();
            
            auto lstm_weights_before = lstm.get_weights();
            
            lstm.eval();
            auto output_before = lstm.forward(inputs);
            
            lstm.train();
            lstm.train_step(inputs, targets, 0.01f);
            
            auto lstm_weights_after = lstm.get_weights();
            
            bool weights_updated = false;
            if (lstm_weights_before.size() == lstm_weights_after.size()) {
                for (size_t layer = 0; layer < lstm_weights_before.size(); ++layer) {
                    if (weights_changed(lstm_weights_before[layer].weight_ih, lstm_weights_after[layer].weight_ih) ||
                        weights_changed(lstm_weights_before[layer].weight_hh, lstm_weights_after[layer].weight_hh) ||
                        weights_changed(lstm_weights_before[layer].bias_ih, lstm_weights_after[layer].bias_ih) ||
                        weights_changed(lstm_weights_before[layer].bias_hh, lstm_weights_after[layer].bias_hh)) {
                        weights_updated = true;
                        break;
                    }
                }
            }
            
            assert_true(weights_updated, "LSTM weights updated during training");
            
            lstm.eval();
            auto output_after = lstm.forward(inputs);
            
            bool output_changed = false;
            for (size_t i = 0; i < output_before.size() && i < output_after.size(); ++i) {
                if (std::abs(output_before[i] - output_after[i]) > EPSILON) {
                    output_changed = true;
                    break;
                }
            }
            
            assert_true(output_changed, "Model output changes after training (indicates parameter updates)");
            
        } catch (const std::exception& e) {
            assert_true(false, "Parameter update test failed: " + std::string(e.what()));
        }
    }

    void test_loss_computation() {
        std::cout << "\n--- Testing Loss Computation ---" << std::endl;
        
        LSTMPredictor lstm(3, 2, 4, 1, 4, true, 42);
        
        std::vector<float> pred1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> target1 = {1.0f, 2.0f, 3.0f};
        float loss1 = lstm.mse_loss(pred1, target1);
        assert_near(loss1, 0.0f, EPSILON, "MSE loss is zero for identical prediction and target");
        
        std::vector<float> pred2 = {0.0f, 0.0f, 0.0f};
        std::vector<float> target2 = {1.0f, 1.0f, 1.0f};
        float loss2 = lstm.mse_loss(pred2, target2);
        assert_near(loss2, 1.0f, TOLERANCE, "MSE loss computed correctly");
        
        std::vector<float> pred3 = {2.0f, 2.0f, 2.0f};
        std::vector<float> target3 = {1.0f, 1.0f, 1.0f};
        float loss3 = lstm.mse_loss(pred3, target3);
        assert_near(loss3, 1.0f, TOLERANCE, "MSE loss symmetric");
        
        std::vector<float> gradient(3);
        lstm.mse_loss_gradient(pred2, target2, gradient);
        std::vector<float> expected_grad = {-2.0f/3.0f, -2.0f/3.0f, -2.0f/3.0f};
        assert_vector_near(gradient, expected_grad, TOLERANCE, "MSE gradient computed correctly");
    }

    void test_gradient_flow() {
        std::cout << "\n--- Testing Gradient Flow ---" << std::endl;
        
        LSTMPredictor lstm(1, 2, 3, 1, 4, true, 42);
        auto [inputs, targets] = generate_test_data(1, 4, 2, 1);
        
        try {
            lstm.train();
            
            float epsilon = 1e-5f;
            float lr = 0.001f;
            
            auto original_weights = lstm.get_weights();
            
            lstm.train_step(inputs, targets, lr);
            auto weights_after = lstm.get_weights();
            
            float max_weight_change = 0.0f;
            if (original_weights.size() == weights_after.size()) {
                for (size_t layer = 0; layer < original_weights.size(); ++layer) {
                    for (size_t i = 0; i < original_weights[layer].weight_ih.size(); ++i) {
                        for (size_t j = 0; j < original_weights[layer].weight_ih[i].size(); ++j) {
                            float change = std::abs(weights_after[layer].weight_ih[i][j] - original_weights[layer].weight_ih[i][j]);
                            max_weight_change = std::max(max_weight_change, change);
                        }
                    }
                    for (size_t i = 0; i < original_weights[layer].weight_hh.size(); ++i) {
                        for (size_t j = 0; j < original_weights[layer].weight_hh[i].size(); ++j) {
                            float change = std::abs(weights_after[layer].weight_hh[i][j] - original_weights[layer].weight_hh[i][j]);
                            max_weight_change = std::max(max_weight_change, change);
                        }
                    }
                }
            }
            
            assert_true(max_weight_change > epsilon, "Gradients flow through network");
            assert_true(max_weight_change < 1.0f, "Weight changes are reasonable magnitude");
            
        } catch (const std::exception& e) {
            assert_true(false, "Gradient flow test failed: " + std::string(e.what()));
        }
    }

    void test_model_persistence() {
        std::cout << "\n--- Testing Model Persistence ---" << std::endl;
        
        LSTMPredictor lstm1(3, 4, 5, 2, 6, true, 42);
        auto [inputs, targets] = generate_test_data(1, 6, 4, 3);
        
        try {
            lstm1.train();
            for (int i = 0; i < 5; ++i) {
                lstm1.train_step(inputs, targets, 0.01f);
            }
            
            std::ofstream save_file("test_model.bin", std::ios::binary);
            lstm1.save_model_state(save_file);
            save_file.close();
            
            LSTMPredictor lstm2(3, 4, 5, 2, 6, true, 123);
            
            std::ifstream load_file("test_model.bin", std::ios::binary);
            lstm2.load_model_state(load_file);
            load_file.close();
            
            lstm1.eval();
            lstm2.eval();
            
            auto output1 = lstm1.forward(inputs);
            auto output2 = lstm2.forward(inputs);
            
            assert_vector_near(output1, output2, TOLERANCE, "Loaded model produces same output");
            
            std::remove("test_model.bin");
            
        } catch (const std::exception& e) {
            assert_true(false, "Model persistence test failed: " + std::string(e.what()));
        }
    }

    void test_state_management() {
        std::cout << "\n--- Testing State Management ---" << std::endl;
        
        LSTMPredictor lstm(2, 3, 4, 1, 5, true, 42);
        auto [inputs, targets] = generate_test_data(1, 5, 3, 2);
        
        try {
            lstm.eval();
            auto output1 = lstm.forward(inputs);
            
            lstm.reset_states();
            auto output2 = lstm.forward(inputs);
            
            assert_vector_near(output1, output2, TOLERANCE, "State reset works correctly");
            
            lstm.train();
            lstm.train_step(inputs, targets, 0.01f);
            
            lstm.eval();
            auto output3 = lstm.forward(inputs);
            
            bool outputs_different = false;
            for (size_t i = 0; i < output1.size(); ++i) {
                if (std::abs(output1[i] - output3[i]) > TOLERANCE) {
                    outputs_different = true;
                    break;
                }
            }
            assert_true(outputs_different, "Training changes model behavior");
            
        } catch (const std::exception& e) {
            assert_true(false, "State management test failed: " + std::string(e.what()));
        }
    }

    void test_error_handling() {
        std::cout << "\n--- Testing Error Handling ---" << std::endl;
        
        try {
            LSTMPredictor lstm(2, 3, 4, 1, 5, true, 42);
            
            std::vector<std::vector<std::vector<float>>> wrong_input(1);
            wrong_input[0].resize(5);
            for (int t = 0; t < 5; ++t) {
                wrong_input[0][t].resize(2);
            }
            
            std::vector<float> targets(2, 0.0f);
            
            try {
                lstm.train();
                lstm.train_step(wrong_input, targets, 0.01f);
                assert_true(false, "Should throw error for wrong input size");
            } catch (const std::exception&) {
                assert_true(true, "Correctly throws error for wrong input size");
            }
            
            std::vector<std::vector<std::vector<float>>> correct_input(1);
            correct_input[0].resize(5);
            for (int t = 0; t < 5; ++t) {
                correct_input[0][t].resize(3);
            }
            
            std::vector<float> wrong_targets(5, 0.0f);
            
            try {
                lstm.train_step(correct_input, wrong_targets, 0.01f);
                assert_true(false, "Should throw error for wrong target size");
            } catch (const std::exception&) {
                assert_true(true, "Correctly throws error for wrong target size");
            }
            
        } catch (const std::exception& e) {
            assert_true(false, "Error handling test setup failed: " + std::string(e.what()));
        }
    }
};

int main() {
    LSTMTestSuite test_suite;
    test_suite.run_all_tests();
    return 0;
}