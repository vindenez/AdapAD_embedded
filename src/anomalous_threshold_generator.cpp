#include "anomalous_threshold_generator.hpp"
#include "config.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lstm_layer, int lstm_unit,
                                                         int lookback_len, int prediction_len)
    : lookback_len(lookback_len), prediction_len(prediction_len) {

    generator = LSTMPredictorFactory::create_predictor(prediction_len, // num_classes
                                                       lookback_len,   // input_size
                                                       lstm_unit,      // hidden_size
                                                       lstm_layer,     // num_layers
                                                       lookback_len,   // lookback_len
                                                       true);          // batch_first

    // Initialize layer cache
    generator->initialize_layer_cache();

    // Pre-allocate vectors for update
    update_input.resize(1);
    update_input[0].resize(1);
    update_input[0][0].resize(lookback_len, 0.0f);
    update_target.resize(1, 0.0f);
    update_pred.resize(1, 0.0f);
    update_output.sequence_output.resize(1);
    update_output.sequence_output[0].resize(1);
    update_output.sequence_output[0][0].resize(prediction_len, 0.0f);
    update_output.final_hidden.resize(lstm_layer);
    update_output.final_cell.resize(lstm_layer);
    for (int i = 0; i < lstm_layer; ++i) {
        update_output.final_hidden[i].resize(lstm_unit, 0.0f);
        update_output.final_cell[i].resize(lstm_unit, 0.0f);
    }
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
AnomalousThresholdGenerator::create_sliding_windows(const std::vector<float> &data) {
    return ::create_sliding_windows(data, lookback_len, prediction_len);
}

float AnomalousThresholdGenerator::generate(const std::vector<float> &prediction_errors,
                                            float minimal_threshold) {

    bool was_training = generator->is_training();
    generator->eval();

    std::vector<std::vector<std::vector<float>>> reshaped_input(1);
    reshaped_input[0].resize(1);
    reshaped_input[0][0] = prediction_errors;

    auto output = generator->forward(reshaped_input);
    auto pred = generator->get_final_prediction(output);

    // Clamp to minimal_threshold
    const auto &config = Config::getInstance();
    float result = std::max(minimal_threshold, pred[0] * config.threshold_multiplier);

    if (was_training) {
        generator->train();
    }

    return result;
}

void AnomalousThresholdGenerator::update(int epoch_update, float lr_update,
                                         const std::vector<float> &past_errors,
                                         float recent_error) {

    // Copy data to pre-allocated input
    std::copy(past_errors.begin(), past_errors.end(), update_input[0][0].begin());

    // Set target
    update_target[0] = recent_error;

    // Training loop with early stopping based on loss progression
    std::vector<float> loss_history;

    generator->train(); // Ensure training mode is enabled

    for (int e = 0; e < epoch_update; ++e) {
        // Perform forward pass to get fresh output for this epoch
        auto output = generator->forward(update_input);

        // Get prediction and compute loss
        auto pred = generator->get_final_prediction(output);
        float current_loss = 0.0f;
        float diff = pred[0] - recent_error;
        current_loss = diff * diff;

        // Early stopping logic
        if (e > 0 && current_loss > loss_history.back()) {
            break;
        }
        loss_history.push_back(current_loss);

        // Train step using fresh forward pass result
        generator->train_step(update_input, update_target, output, lr_update);
    }
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<float>>
AnomalousThresholdGenerator::train(int epoch, float lr, const std::vector<float> &data2learn) {
    if (data2learn.size() < lookback_len + prediction_len) {
        throw std::runtime_error("Not enough data for generator training");
    }

    auto windows = create_sliding_windows(data2learn);
    generator->train();

    for (int e = 0; e < epoch; ++e) {
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < windows.first.size(); ++i) {

            std::vector<std::vector<std::vector<float>>> reshaped_input(1);
            reshaped_input[0].resize(1);
            reshaped_input[0][0] = windows.first[i];

            std::vector<float> target{windows.second[i]};

            auto output = generator->forward(reshaped_input);
            auto pred = generator->get_final_prediction(output);

            float diff = pred[0] - target[0];
            epoch_loss += diff * diff;

            generator->train_step(reshaped_input, target, output, lr);
        }

        // Report progress
        if ((e + 1) % 100 == 0) {
            float avg_loss = epoch_loss / static_cast<float>(windows.first.size());
            std::cout << "Generator Epoch " << (e + 1) << "/" << epoch
                      << ", Average Loss: " << avg_loss << std::endl;
        }
    }

    generator->learn();

    // Return processed windows in the expected format
    std::vector<std::vector<std::vector<float>>> x3d;
    for (const auto &window : windows.first) {
        std::vector<std::vector<std::vector<float>>> input_tensor(1);
        input_tensor[0].resize(1);
        input_tensor[0][0] = window;
        x3d.push_back(input_tensor[0]);
    }

    return {x3d, windows.second};
}

void AnomalousThresholdGenerator::save_weights(std::ofstream &file) {
    if (generator) {
        generator->save_weights(file);
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::save_biases(std::ofstream &file) {
    if (generator) {
        generator->save_biases(file);
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::load_weights(std::ifstream &file) {
    if (generator) {
        generator->load_weights(file);
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::load_biases(std::ifstream &file) {
    if (generator) {
        generator->load_biases(file);
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::save_layer_cache(std::ofstream &file) const {
    if (generator) {
        generator->save_layer_cache(file);
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::load_layer_cache(std::ifstream &file) {
    if (generator) {
        generator->load_layer_cache(file);
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::initialize_layer_cache() {
    if (generator) {
        generator->initialize_layer_cache();
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::reset_states() {
    if (generator) {
        generator->reset_states();
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}

void AnomalousThresholdGenerator::clear_update_state() {
    if (generator) {
        generator->clear_update_state();
    } else {
        throw std::runtime_error("Generator not initialized");
    }
}
