#define TESTING
#include "config.hpp"
#include "lstm_predictor.hpp"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

#ifdef DEBUG
#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x)                                                                             \
    do {                                                                                           \
    } while (0)
#endif

class LSTMPredictorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a small LSTM for testing
        num_classes = 1; // Single output for regression
        input_size = 1;  // Single feature
        hidden_size = 4; // Smaller hidden size
        num_layers = 1;  // Single layer
        seq_length = 3;  // Shorter sequence

        lstm = std::make_unique<LSTMPredictor>(num_classes, input_size, hidden_size, num_layers,
                                               seq_length);
    }

    // Helper function to create sample input
    std::vector<std::vector<std::vector<float>>> create_sample_input(size_t batch_size = 1,
                                                                     size_t seq_len = 5) {
        std::vector<std::vector<std::vector<float>>> input(
            batch_size,
            std::vector<std::vector<float>>(seq_len, std::vector<float>(input_size, 0.1f)));
        return input;
    }

    // Helper function to check if values are close
    bool is_close(float a, float b, float tolerance = 1e-5f) { return std::abs(a - b) < tolerance; }

    // Helper function for gradient computation
    std::vector<LSTMPredictor::LSTMGradients>
    compute_numerical_gradients(LSTMPredictor *predictor,
                                const std::vector<std::vector<std::vector<float>>> &x,
                                const std::vector<float> &target, float epsilon) {

        // Compute base loss once at the start
        predictor->reset_states();
        auto base_output = predictor->forward(x);
        auto base_pred = predictor->get_final_prediction(base_output);
        float base_loss = predictor->compute_loss(base_pred, target);
        DEBUG_PRINT(("Base loss: %f, Base prediction: %f, Target: %f\n", base_loss, base_pred[0],
                     target[0]));

        auto original_weights = predictor->get_weights();
        auto numerical_grads = predictor->get_last_gradients(); // For structure

        // For each layer
        for (int layer = 0; layer < predictor->get_num_layers(); ++layer) {
            DEBUG_PRINT(("Computing gradients for layer %d\n", layer));

            // Input-hidden weights
            DEBUG_PRINT(("Computing input-hidden weight gradients\n"));
            for (size_t i = 0; i < numerical_grads[layer].weight_ih_grad.size(); ++i) {
                for (size_t j = 0; j < numerical_grads[layer].weight_ih_grad[i].size(); ++j) {
                    float original_weight = original_weights[layer].weight_ih[i][j];
                    DEBUG_PRINT(("Computing gradient for weight_ih[%zu][%zu], original "
                                 "value: %f\n",
                                 i, j, original_weight));

                    // Compute f(x + epsilon)
                    auto perturbed_weights = original_weights;
                    perturbed_weights[layer].weight_ih[i][j] = original_weight + epsilon;
                    predictor->set_weights(perturbed_weights);
                    predictor->reset_states();
                    auto output_plus = predictor->forward(x);
                    auto pred_plus = predictor->get_final_prediction(output_plus);
                    float loss_plus = predictor->compute_loss(pred_plus, target);

                    // Compute f(x - epsilon)
                    perturbed_weights[layer].weight_ih[i][j] = original_weight - epsilon;
                    predictor->set_weights(perturbed_weights);
                    predictor->reset_states();
                    auto output_minus = predictor->forward(x);
                    auto pred_minus = predictor->get_final_prediction(output_minus);
                    float loss_minus = predictor->compute_loss(pred_minus, target);

                    DEBUG_PRINT(("  weight_ih[%zu][%zu] results:\n", i, j));
                    DEBUG_PRINT(("    loss_plus=%f (pred=%f)\n", loss_plus, pred_plus[0]));
                    DEBUG_PRINT(("    loss_minus=%f (pred=%f)\n", loss_minus, pred_minus[0]));
                    DEBUG_PRINT(("    loss_diff=%f\n", loss_plus - loss_minus));

                    numerical_grads[layer].weight_ih_grad[i][j] =
                        (loss_plus - loss_minus) / (2 * epsilon);
                }
            }

            // Hidden-hidden weights
            DEBUG_PRINT(("Computing hidden-hidden weight gradients\n"));
            for (size_t i = 0; i < numerical_grads[layer].weight_hh_grad.size(); ++i) {
                for (size_t j = 0; j < numerical_grads[layer].weight_hh_grad[i].size(); ++j) {
                    float original_weight = original_weights[layer].weight_hh[i][j];
                    DEBUG_PRINT(("Computing gradient for weight_hh[%zu][%zu], original "
                                 "value: %f\n",
                                 i, j, original_weight));

                    auto weights_plus = original_weights;
                    weights_plus[layer].weight_hh[i][j] += epsilon;
                    predictor->set_weights(weights_plus);
                    predictor->reset_states(); // Reset states before forward pass
                    auto output_plus = predictor->forward(x);
                    auto pred_plus = predictor->get_final_prediction(output_plus);
                    float loss_plus = predictor->compute_loss(pred_plus, target);

                    auto weights_minus = original_weights;
                    weights_minus[layer].weight_hh[i][j] -= epsilon;
                    predictor->set_weights(weights_minus);
                    predictor->reset_states(); // Reset states before forward pass
                    auto output_minus = predictor->forward(x);
                    auto pred_minus = predictor->get_final_prediction(output_minus);
                    float loss_minus = predictor->compute_loss(pred_minus, target);

                    DEBUG_PRINT(("  weight_hh[%zu][%zu] results:\n", i, j));
                    DEBUG_PRINT(("    loss_plus=%f (pred=%f)\n", loss_plus, pred_plus[0]));
                    DEBUG_PRINT(("    loss_minus=%f (pred=%f)\n", loss_minus, pred_minus[0]));
                    DEBUG_PRINT(("    loss_diff=%f\n", loss_plus - loss_minus));

                    numerical_grads[layer].weight_hh_grad[i][j] =
                        (loss_plus - loss_minus) / (2 * epsilon);
                }
            }

            // Input bias
            DEBUG_PRINT(("Computing input bias gradients\n"));
            for (size_t i = 0; i < numerical_grads[layer].bias_ih_grad.size(); ++i) {
                float original_bias = original_weights[layer].bias_ih[i];
                DEBUG_PRINT(("Computing gradient for bias_ih[%zu], original value: %f\n", i,
                             original_bias));

                auto weights_plus = original_weights;
                weights_plus[layer].bias_ih[i] += epsilon;
                predictor->set_weights(weights_plus);
                predictor->reset_states(); // Reset states before forward pass
                auto output_plus = predictor->forward(x);
                auto pred_plus = predictor->get_final_prediction(output_plus);
                float loss_plus = predictor->compute_loss(pred_plus, target);

                auto weights_minus = original_weights;
                weights_minus[layer].bias_ih[i] -= epsilon;
                predictor->set_weights(weights_minus);
                predictor->reset_states(); // Reset states before forward pass
                auto output_minus = predictor->forward(x);
                auto pred_minus = predictor->get_final_prediction(output_minus);
                float loss_minus = predictor->compute_loss(pred_minus, target);

                DEBUG_PRINT(("  bias_ih[%zu] results:\n", i));
                DEBUG_PRINT(("    loss_plus=%f (pred=%f)\n", loss_plus, pred_plus[0]));
                DEBUG_PRINT(("    loss_minus=%f (pred=%f)\n", loss_minus, pred_minus[0]));
                DEBUG_PRINT(("    loss_diff=%f\n", loss_plus - loss_minus));

                numerical_grads[layer].bias_ih_grad[i] = (loss_plus - loss_minus) / (2 * epsilon);
            }

            // Hidden bias
            DEBUG_PRINT(("Computing hidden bias gradients\n"));
            for (size_t i = 0; i < numerical_grads[layer].bias_hh_grad.size(); ++i) {
                DEBUG_PRINT(("Computing numerical gradients for bias_hh[%zu]:\n",
                             i)); // cell gate offset
                float original_bias = original_weights[layer].bias_hh[i];
                DEBUG_PRINT(("Original bias value: %f\n", original_bias));
                DEBUG_PRINT(("Computing gradient for bias_hh[%zu], original value: %f\n", i,
                             original_bias));

                auto weights_plus = original_weights;
                weights_plus[layer].bias_hh[i] += epsilon;
                predictor->set_weights(weights_plus);
                predictor->reset_states(); // Reset states before forward pass
                auto output_plus = predictor->forward(x);
                auto pred_plus = predictor->get_final_prediction(output_plus);
                float loss_plus = predictor->compute_loss(pred_plus, target);

                auto weights_minus = original_weights;
                weights_minus[layer].bias_hh[i] -= epsilon;
                predictor->set_weights(weights_minus);
                predictor->reset_states(); // Reset states before forward pass
                auto output_minus = predictor->forward(x);
                auto pred_minus = predictor->get_final_prediction(output_minus);
                float loss_minus = predictor->compute_loss(pred_minus, target);

                DEBUG_PRINT(("  bias_hh[%zu] results:\n", i));
                DEBUG_PRINT(("    loss_plus=%f (pred=%f)\n", loss_plus, pred_plus[0]));
                DEBUG_PRINT(("    loss_minus=%f (pred=%f)\n", loss_minus, pred_minus[0]));
                DEBUG_PRINT(("    loss_diff=%f\n", loss_plus - loss_minus));

                numerical_grads[layer].bias_hh_grad[i] = (loss_plus - loss_minus) / (2 * epsilon);
            }
        }

        DEBUG_PRINT(("Numerical gradient computation complete\n"));
        predictor->set_weights(original_weights);
        predictor->reset_states();
        return numerical_grads;
    }

    // Helper function to compare gradients with detailed logging
    float compare_gradients(const std::vector<LSTMPredictor::LSTMGradients> &analytical,
                            const std::vector<LSTMPredictor::LSTMGradients> &numerical) {

        float max_diff = 0.0f;
        for (size_t layer = 0; layer < analytical.size(); ++layer) {
            // Compare input-hidden weight gradients
            for (size_t i = 0; i < analytical[layer].weight_ih_grad.size(); ++i) {
                for (size_t j = 0; j < analytical[layer].weight_ih_grad[i].size(); ++j) {
                    float diff = std::abs(analytical[layer].weight_ih_grad[i][j] -
                                          numerical[layer].weight_ih_grad[i][j]);
                    if (diff > 1e-3) {
                        DEBUG_PRINT(("Large gradient diff at layer %zu, weight_ih[%zu][%zu]:\n",
                                     layer, i, j));
                        DEBUG_PRINT(("  analytical=%f\n  numerical=%f\n  diff=%f\n",
                                     analytical[layer].weight_ih_grad[i][j],
                                     numerical[layer].weight_ih_grad[i][j], diff));
                    }
                    max_diff = std::max(max_diff, diff);
                }
            }

            // Compare hidden-hidden weight gradients
            for (size_t i = 0; i < analytical[layer].weight_hh_grad.size(); ++i) {
                for (size_t j = 0; j < analytical[layer].weight_hh_grad[i].size(); ++j) {
                    float diff = std::abs(analytical[layer].weight_hh_grad[i][j] -
                                          numerical[layer].weight_hh_grad[i][j]);
                    if (diff > 1e-3) {
                        DEBUG_PRINT(("Large gradient diff at layer %zu, weight_hh[%zu][%zu]:\n",
                                     layer, i, j));
                        DEBUG_PRINT(("  analytical=%f\n  numerical=%f\n  diff=%f\n",
                                     analytical[layer].weight_hh_grad[i][j],
                                     numerical[layer].weight_hh_grad[i][j], diff));
                    }
                    max_diff = std::max(max_diff, diff);
                }
            }

            // Compare input bias gradients
            for (size_t i = 0; i < analytical[layer].bias_ih_grad.size(); ++i) {
                float diff =
                    std::abs(analytical[layer].bias_ih_grad[i] - numerical[layer].bias_ih_grad[i]);
                if (diff > 1e-3) {
                    DEBUG_PRINT(("Large gradient diff at layer %zu, bias_ih[%zu]:\n", layer, i));
                    DEBUG_PRINT(("  analytical=%f\n  numerical=%f\n  diff=%f\n",
                                 analytical[layer].bias_ih_grad[i],
                                 numerical[layer].bias_ih_grad[i], diff));
                }
                max_diff = std::max(max_diff, diff);
            }

            // Compare hidden bias gradients
            for (size_t i = 0; i < analytical[layer].bias_hh_grad.size(); ++i) {
                float diff =
                    std::abs(analytical[layer].bias_hh_grad[i] - numerical[layer].bias_hh_grad[i]);
                if (diff > 1e-3) {
                    DEBUG_PRINT(("Large gradient diff at layer %zu, bias_hh[%zu]:\n", layer, i));
                    DEBUG_PRINT(("  analytical=%f\n  numerical=%f\n  diff=%f\n",
                                 analytical[layer].bias_hh_grad[i],
                                 numerical[layer].bias_hh_grad[i], diff));
                }
                max_diff = std::max(max_diff, diff);
            }
        }

        // Summary of comparison
        if (max_diff > 1e-3) {
            DEBUG_PRINT(("Gradient check summary:\n"));
            DEBUG_PRINT(("  Maximum difference: %f\n", max_diff));
            DEBUG_PRINT(("  Number of layers checked: %zu\n", analytical.size()));
            DEBUG_PRINT(("  Tolerance threshold: 1e-3\n"));
        } else {
            DEBUG_PRINT(("All gradients match within tolerance (max diff: %f)\n", max_diff));
        }

        return max_diff;
    }

    int num_classes;
    int input_size;
    int hidden_size;
    int num_layers;
    int seq_length;
    std::unique_ptr<LSTMPredictor> lstm;
};

// Test forward pass dimensions
TEST_F(LSTMPredictorTest, ForwardPassDimensions) {
    auto input = create_sample_input(2, 5); // batch_size=2, seq_len=5
    auto output = lstm->forward(input);

    // Check sequence output dimensions
    EXPECT_EQ(output.sequence_output.size(), 2);    // batch_size
    EXPECT_EQ(output.sequence_output[0].size(), 5); // seq_len
    EXPECT_EQ(output.sequence_output[0][0].size(), hidden_size);

    // Check final states dimensions
    EXPECT_EQ(output.final_hidden.size(), num_layers);
    EXPECT_EQ(output.final_hidden[0].size(), hidden_size);
    EXPECT_EQ(output.final_cell.size(), num_layers);
    EXPECT_EQ(output.final_cell[0].size(), hidden_size);
}

// Test state reset
TEST_F(LSTMPredictorTest, StateReset) {
    lstm->reset_states();
    auto input = create_sample_input();
    auto output1 = lstm->forward(input);

    lstm->reset_states();
    auto output2 = lstm->forward(input);

    // Check if outputs are the same after reset
    for (size_t i = 0; i < output1.final_hidden[0].size(); ++i) {
        EXPECT_TRUE(is_close(output1.final_hidden[0][i], output2.final_hidden[0][i]));
    }
}

// Test training step
TEST_F(LSTMPredictorTest, TrainingStep) {
    DEBUG_PRINT(("Creating sample input\n"));
    auto input = create_sample_input(1, 1); // batch_size=1, seq_len=1
    std::vector<float> target = {1.0f, 0.0f};

    DEBUG_PRINT(("Running initial forward pass\n"));
    auto initial_output = lstm->forward(input);
    auto initial_pred = lstm->get_final_prediction(initial_output);
    float initial_loss = lstm->compute_loss(initial_pred, target);

    DEBUG_PRINT(("Initial loss: %f\n", initial_loss));

    DEBUG_PRINT(("Running training step\n"));
    lstm->train_step(input, target, 0.01f);

    DEBUG_PRINT(("Running final forward pass\n"));
    auto final_output = lstm->forward(input);
    auto final_pred = lstm->get_final_prediction(final_output);
    float final_loss = lstm->compute_loss(final_pred, target);

    DEBUG_PRINT(("Final loss: %f\n", final_loss));

    EXPECT_LT(final_loss, initial_loss);
}

// Test gradient computation
TEST_F(LSTMPredictorTest, GradientCheck) {
    // Create test input
    auto input = create_sample_input(1, seq_length);
    std::vector<float> target{0.5f}; // Single target value

    // Store original weights
    auto original_weights = lstm->get_weights();

    // Compute analytical gradients
    float learning_rate = 0.01f;
    lstm->train_step(input, target, learning_rate);
    auto analytical_grads = lstm->get_last_gradients();

    // Restore original weights
    lstm->set_weights(original_weights);

    // Compute numerical gradients
    float epsilon = 1e-5;
    auto numerical_grads = compute_numerical_gradients(lstm.get(), input, target, epsilon);

    // Compare gradients
    float max_diff = compare_gradients(analytical_grads, numerical_grads);

    EXPECT_LT(max_diff, 1e-5);
}

// Test batch processing
TEST_F(LSTMPredictorTest, BatchProcessing) {
    auto input = create_sample_input(3, 5); // batch_size=3
    auto output = lstm->forward(input);

    // Check if batch processing works
    EXPECT_EQ(output.sequence_output.size(), 3);
    for (const auto &batch : output.sequence_output) {
        EXPECT_EQ(batch.size(), 5);
        for (const auto &seq : batch) {
            EXPECT_EQ(seq.size(), hidden_size);
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}