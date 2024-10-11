#include "anomalous_threshold_generator.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Constructor to initialize using weights and biases for each gate
AnomalousThresholdGenerator::AnomalousThresholdGenerator(
    const std::vector<std::vector<float>>& weight_ih_input,
    const std::vector<std::vector<float>>& weight_hh_input,
    const std::vector<float>& bias_ih_input,
    const std::vector<float>& bias_hh_input,
    const std::vector<std::vector<float>>& weight_ih_forget,
    const std::vector<std::vector<float>>& weight_hh_forget,
    const std::vector<float>& bias_ih_forget,
    const std::vector<float>& bias_hh_forget,
    const std::vector<std::vector<float>>& weight_ih_output,
    const std::vector<std::vector<float>>& weight_hh_output,
    const std::vector<float>& bias_ih_output,
    const std::vector<float>& bias_hh_output,
    const std::vector<std::vector<float>>& weight_ih_cell,
    const std::vector<std::vector<float>>& weight_hh_cell,
    const std::vector<float>& bias_ih_cell,
    const std::vector<float>& bias_hh_cell)
    : generator(weight_ih_input, weight_hh_input, bias_ih_input, bias_hh_input,
                weight_ih_forget, weight_hh_forget, bias_ih_forget, bias_hh_forget,
                weight_ih_output, weight_hh_output, bias_ih_output, bias_hh_output,
                weight_ih_cell, weight_hh_cell, bias_ih_cell, bias_hh_cell),
      h(generator.get_hidden_size(), 0.0f),
      c(generator.get_hidden_size(), 0.0f) {
    
    // Initialize other members
    lookback_len = weight_ih_input[0].size();
    prediction_len = 1; // Assuming single step prediction
    lower_bound = 0.0f; // Set default values or add parameters to constructor
    upper_bound = 1.0f; // Set default values or add parameters to constructor
}

// Constructor to initialize using hyperparameters
AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lookback_len, int prediction_len, float lower_bound, float upper_bound)
    : lookback_len(lookback_len), prediction_len(prediction_len), lower_bound(lower_bound), upper_bound(upper_bound),
      generator(lookback_len, prediction_len, 1, lookback_len),
      h(generator.get_hidden_size(), 0.0f),
      c(generator.get_hidden_size(), 0.0f) {}

// Update function for feed-forward adaptation of the generator
void AnomalousThresholdGenerator::update(int num_epochs, float learning_rate, const std::vector<float>& past_errors) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward pass
        auto [output, new_h, new_c] = generator.forward(past_errors, h, c);

        // Compute loss gradient
        std::vector<float> doutput(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            doutput[i] = 2.0f * (output[i] - past_errors.back()) / output.size();  // MSE loss gradient
        }

        // Backward pass
        auto [dh, dc, dw_ih, dw_hh, db_ih, db_hh] = backward_step(past_errors, h, c, doutput);

        // Update parameters
        update_parameters(dw_ih, dw_hh, db_ih, db_hh, learning_rate);

        // Update hidden state and cell state
        h = new_h;
        c = new_c;

        // Compute and print loss for monitoring
        float loss = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            loss += (output[i] - past_errors.back()) * (output[i] - past_errors.back());
        }
        loss /= output.size();
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss << std::endl;
    }
}

std::tuple<std::vector<float>, std::vector<float>, 
           std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<float>, std::vector<float>>
AnomalousThresholdGenerator::backward_step(const std::vector<float>& input,
                                           const std::vector<float>& h_prev,
                                           const std::vector<float>& c_prev,
                                           const std::vector<float>& doutput) {
    // Retrieve current weights and biases
    auto weight_ih = generator.get_weight_ih_input();
    auto weight_hh = generator.get_weight_hh_input();
    auto bias_ih = generator.get_bias_ih_input();
    auto bias_hh = generator.get_bias_hh_input();

    // Initialize gradients
    std::vector<std::vector<float>> dw_ih(weight_ih.size(), std::vector<float>(weight_ih[0].size(), 0.0f));
    std::vector<std::vector<float>> dw_hh(weight_hh.size(), std::vector<float>(weight_hh[0].size(), 0.0f));
    std::vector<float> db_ih(bias_ih.size(), 0.0f);
    std::vector<float> db_hh(bias_hh.size(), 0.0f);

    // Compute gradients for the output gate
    std::vector<float> dh = doutput;
    std::vector<float> dc(c_prev.size(), 0.0f);

    // Backward pass through time steps
    for (int t = input.size() - 1; t >= 0; --t) {
        // Compute gate activations
        auto [i_t, f_t, o_t, g_t, c_t, h_t] = generator.forward_step(input[t], h_prev, c_prev);

        // Compute gradients for output gate
        std::vector<float> tanh_c_t(c_t.size());
        for (size_t i = 0; i < c_t.size(); ++i) {
            tanh_c_t[i] = tanh_func(c_t[i]);
        }
        std::vector<float> do_t = elementwise_mul(dh, tanh_c_t);
        for (size_t i = 0; i < do_t.size(); ++i) {
            do_t[i] *= o_t[i] * (1 - o_t[i]);  // Derivative of sigmoid
        }

        // Compute gradients for cell state
        std::vector<float> d_tanh_c_t(c_t.size());
        for (size_t i = 0; i < c_t.size(); ++i) {
            d_tanh_c_t[i] = d_tanh_func(c_t[i]);
        }
        std::vector<float> dc_t = elementwise_add(dc, elementwise_mul(dh, elementwise_mul(o_t, d_tanh_c_t)));

        // Compute gradients for input gate
        std::vector<float> di_t = elementwise_mul(dc_t, g_t);
        for (size_t i = 0; i < di_t.size(); ++i) {
            di_t[i] *= i_t[i] * (1 - i_t[i]);  // Derivative of sigmoid
        }

        // Compute gradients for forget gate
        std::vector<float> df_t = elementwise_mul(dc_t, c_prev);
        for (size_t i = 0; i < df_t.size(); ++i) {
            df_t[i] *= f_t[i] * (1 - f_t[i]);  // Derivative of sigmoid
        }

        // Compute gradients for g
        std::vector<float> dg_t = elementwise_mul(dc_t, i_t);
        for (size_t i = 0; i < dg_t.size(); ++i) {
            dg_t[i] *= (1 - g_t[i] * g_t[i]);  // Derivative of tanh
        }

        // Compute gradients for weights and biases
        std::vector<float> dxh_t = elementwise_add(elementwise_add(di_t, df_t), elementwise_add(do_t, dg_t));
        
        // Update gradients for this time step
        for (size_t i = 0; i < dw_ih.size(); ++i) {
            dw_ih[i][0] += dxh_t[i] * input[t];
            for (size_t j = 0; j < dw_hh[i].size(); ++j) {
                dw_hh[i][j] += dxh_t[i] * h_prev[j];
            }
            db_ih[i] += dxh_t[i];
            db_hh[i] += dxh_t[i];
        }

        // Update dh for the next time step
        dh = matrix_vector_mul(weight_hh, dxh_t);
    }

    return {dh, dc, dw_ih, dw_hh, db_ih, db_hh};
}

void AnomalousThresholdGenerator::update_parameters(const std::vector<std::vector<float>>& dw_ih,
                                                    const std::vector<std::vector<float>>& dw_hh,
                                                    const std::vector<float>& db_ih,
                                                    const std::vector<float>& db_hh,
                                                    float learning_rate) {
    auto weight_ih = generator.get_weight_ih_input();
    auto weight_hh = generator.get_weight_hh_input();
    auto bias_ih = generator.get_bias_ih_input();
    auto bias_hh = generator.get_bias_hh_input();

    // Update weights
    for (size_t i = 0; i < weight_ih.size(); ++i) {
        for (size_t j = 0; j < weight_ih[i].size(); ++j) {
            weight_ih[i][j] -= learning_rate * dw_ih[i][j];
            weight_hh[i][j] -= learning_rate * dw_hh[i][j];
        }
    }

    // Update biases
    for (size_t i = 0; i < bias_ih.size(); ++i) {
        bias_ih[i] -= learning_rate * db_ih[i];
        bias_hh[i] -= learning_rate * db_hh[i];
    }

    // Set updated weights and biases
    generator.set_weight_ih_input(weight_ih);
    generator.set_weight_hh_input(weight_hh);
    generator.set_bias_ih_input(bias_ih);
    generator.set_bias_hh_input(bias_hh);
}

float AnomalousThresholdGenerator::generate(const std::vector<float>& prediction_errors, float minimal_threshold) {
    if (prediction_errors.empty() || prediction_errors.size() != lookback_len) {
        std::cerr << "Error: Invalid prediction_errors size in generate(). Expected: " << lookback_len 
                  << ", Got: " << prediction_errors.size() << std::endl;
        return minimal_threshold;
    }

    std::vector<float> input(lookback_len, 0.0f);
    std::copy(prediction_errors.end() - lookback_len, prediction_errors.end(), input.begin());

    std::tie(output, h, c) = generator.forward(input, h, c);

    if (output.empty()) {
        std::cerr << "Error: output is empty after generator forward pass in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }

    float threshold = std::max(minimal_threshold, output[0]);
    std::cout << "Generated threshold: " << threshold << std::endl;

    return threshold;
}

float AnomalousThresholdGenerator::generate_threshold(const std::vector<float>& new_input) {
    if (new_input.size() != lookback_len) {
        throw std::invalid_argument("Input size does not match lookback length");
    }

    // Forward pass through the LSTM
    std::tie(output, h, c) = generator.forward(new_input, h, c);

    // Calculate the mean of the output
    float mean = std::accumulate(output.begin(), output.end(), 0.0f) / output.size();

    // Clamp the mean between lower_bound and upper_bound
    return std::clamp(mean, lower_bound, upper_bound);
}

std::vector<float> AnomalousThresholdGenerator::generate_thresholds(const std::vector<std::vector<float>>& input_sequence) {
    std::vector<float> thresholds;
    thresholds.reserve(input_sequence.size());

    for (const auto& input : input_sequence) {
        thresholds.push_back(generate_threshold(input));
    }

    return thresholds;
}