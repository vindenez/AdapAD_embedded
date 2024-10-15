#include "anomalous_threshold_generator.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include "config.hpp"
#include <random>
#include <limits>


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
    // Add input_size and hidden_size parameters
    : generator(weight_ih_input, weight_hh_input, bias_ih_input, bias_hh_input,
                weight_ih_forget, weight_hh_forget, bias_ih_forget, bias_hh_forget,
                weight_ih_output, weight_hh_output, bias_ih_output, bias_hh_output,
                weight_ih_cell, weight_hh_cell, bias_ih_cell, bias_hh_cell,
                /* Add these two parameters: */
                weight_ih_input[0].size(),    // input_size
                weight_ih_input.size())       // hidden_size
    , h(generator.get_hidden_size(), 0.0f)
    , c(generator.get_hidden_size(), 0.0f) {
    // Initialize other members
    lookback_len = weight_ih_input[0].size();
    prediction_len = 1; // Assuming single step prediction
    lower_bound = 0.0f; // Set default values or add parameters to constructor
    upper_bound = 1.0f; // Set default values or add parameters to constructor
}


// Constructor to initialize using hyperparameters
AnomalousThresholdGenerator::AnomalousThresholdGenerator(int lookback_len, int prediction_len, float lower_bound, float upper_bound)
    : lookback_len(lookback_len),
      prediction_len(prediction_len),
      lower_bound(lower_bound),
      upper_bound(upper_bound),
      generator(
          lookback_len,               // input_size
          config::LSTM_size,          // hidden_size
          config::LSTM_size_layer,    // num_layers
          lookback_len                // lookback_len (sequence length)
      ),
      h(config::LSTM_size, 0.0f),
      c(config::LSTM_size, 0.0f),
      t(0) {
    init_adam_parameters();
}


void AnomalousThresholdGenerator::init_adam_parameters() {
    beta1 = 0.9f;
    beta2 = 0.999f;
    epsilon = 1e-8f;
    t = 0;

    auto weight_ih = generator.get_weight_ih_input();
    auto weight_hh = generator.get_weight_hh_input();
    auto bias_ih = generator.get_bias_ih_input();
    auto bias_hh = generator.get_bias_hh_input();

    m_w_ih = std::vector<std::vector<float>>(weight_ih.size(), std::vector<float>(weight_ih[0].size(), 0.0f));
    m_w_hh = std::vector<std::vector<float>>(weight_hh.size(), std::vector<float>(weight_hh[0].size(), 0.0f));
    m_b_ih = std::vector<float>(bias_ih.size(), 0.0f);
    m_b_hh = std::vector<float>(bias_hh.size(), 0.0f);

    v_w_ih = std::vector<std::vector<float>>(weight_ih.size(), std::vector<float>(weight_ih[0].size(), 0.0f));
    v_w_hh = std::vector<std::vector<float>>(weight_hh.size(), std::vector<float>(weight_hh[0].size(), 0.0f));
    v_b_ih = std::vector<float>(bias_ih.size(), 0.0f);
    v_b_hh = std::vector<float>(bias_hh.size(), 0.0f);
}

// Update function for feed-forward adaptation of the generator
void AnomalousThresholdGenerator::update(float learning_rate, const std::vector<float>& past_errors) {
    const float epsilon = 1e-8;  // Small value to prevent division by zero
    
    // Forward pass
    auto [output, new_h, new_c] = generator.forward(past_errors, h, c);

    // Compute loss gradient
    std::vector<float> doutput(output.size());
    double total_loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double error = output[i] - past_errors.back();
        doutput[i] = 2.0f * static_cast<float>(error) / (output.size() + epsilon);  // MSE loss gradient
        total_loss += error * error;
    }
    total_loss /= (output.size() + epsilon);

    std::cout << "Update - Loss: " << total_loss << std::endl;

    // Backward pass
    auto [dh, dc, dw_ih, dw_hh, db_ih, db_hh] = backward_step(past_errors, h, c, doutput);

    // Update parameters using Adam
    update_parameters_adam(dw_ih, dw_hh, db_ih, db_hh, learning_rate);

    // Update hidden state and cell state
    h = new_h;
    c = new_c;

    std::cout << "Update complete, Loss: " << total_loss << std::endl;
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


float AnomalousThresholdGenerator::generate(const std::vector<float>& prediction_errors, float minimal_threshold) {
    if (prediction_errors.empty() || prediction_errors.size() != lookback_len) {
        std::cerr << "Error: Invalid prediction_errors size in generate(). Expected: " << lookback_len 
                  << ", Got: " << prediction_errors.size() << std::endl;
        return minimal_threshold;
    }

    std::vector<float> input(lookback_len, 0.0f);
    std::copy(prediction_errors.end() - lookback_len, prediction_errors.end(), input.begin());

    auto [output, new_h, new_c] = generator.forward(input, h, c);

    if (output.empty()) {
        std::cerr << "Error: output is empty after generator forward pass in generate(). Returning minimal_threshold." << std::endl;
        return minimal_threshold;
    }

    float threshold = output[0];
    threshold = std::max(minimal_threshold, threshold);
    std::cout << "Generated threshold: " << threshold << " (minimal: " << minimal_threshold << ")" << std::endl;

    // Update hidden and cell states
    h = new_h;
    c = new_c;

    return threshold;
}

float AnomalousThresholdGenerator::generate_threshold(const std::vector<float>& new_input) {
    if (new_input.size() != lookback_len) {
        throw std::invalid_argument("Input size does not match lookback length");
    }

    // Forward pass through the LSTM
    std::tie(output, h, c) = generator.forward(new_input, h, c);

    // Calculate the mean of the output
    float mean_output = std::accumulate(output.begin(), output.end(), 0.0f) / output.size();

    // Apply sigmoid to get the final threshold
    float threshold = 1.0f / (1.0f + std::exp(-mean_output));

    return threshold;
}

std::vector<float> AnomalousThresholdGenerator::generate_thresholds(const std::vector<std::vector<float>>& input_sequence) {
    std::vector<float> thresholds;
    thresholds.reserve(input_sequence.size());

    for (const auto& input : input_sequence) {
        thresholds.push_back(generate_threshold(input));
    }

    return thresholds;
}

void AnomalousThresholdGenerator::update_parameters_adam(
    const std::vector<std::vector<float>>& dw_ih,
    const std::vector<std::vector<float>>& dw_hh,
    const std::vector<float>& db_ih,
    const std::vector<float>& db_hh,
    float learning_rate) {
    
    t++;
    float alpha_t = learning_rate * std::sqrt(1 - std::pow(beta2, t)) / (1 - std::pow(beta1, t));

    auto weight_ih = generator.get_weight_ih_input();
    auto weight_hh = generator.get_weight_hh_input();
    auto bias_ih = generator.get_bias_ih_input();
    auto bias_hh = generator.get_bias_hh_input();

    // Update weights
    for (size_t i = 0; i < weight_ih.size(); ++i) {
        for (size_t j = 0; j < weight_ih[i].size(); ++j) {
            // Update biased first moment estimate
            m_w_ih[i][j] = beta1 * m_w_ih[i][j] + (1 - beta1) * dw_ih[i][j];
            m_w_hh[i][j] = beta1 * m_w_hh[i][j] + (1 - beta1) * dw_hh[i][j];

            // Update biased second raw moment estimate
            v_w_ih[i][j] = beta2 * v_w_ih[i][j] + (1 - beta2) * dw_ih[i][j] * dw_ih[i][j];
            v_w_hh[i][j] = beta2 * v_w_hh[i][j] + (1 - beta2) * dw_hh[i][j] * dw_hh[i][j];

            // Compute bias-corrected first moment estimate
            float m_hat_w_ih = m_w_ih[i][j] / (1 - std::pow(beta1, t));
            float m_hat_w_hh = m_w_hh[i][j] / (1 - std::pow(beta1, t));

            // Compute bias-corrected second raw moment estimate
            float v_hat_w_ih = v_w_ih[i][j] / (1 - std::pow(beta2, t));
            float v_hat_w_hh = v_w_hh[i][j] / (1 - std::pow(beta2, t));

            // Update parameters
            weight_ih[i][j] -= alpha_t * m_hat_w_ih / (std::sqrt(v_hat_w_ih) + epsilon);
            weight_hh[i][j] -= alpha_t * m_hat_w_hh / (std::sqrt(v_hat_w_hh) + epsilon);
        }
    }

    // Update biases
    for (size_t i = 0; i < bias_ih.size(); ++i) {
        // Update biased first moment estimate
        m_b_ih[i] = beta1 * m_b_ih[i] + (1 - beta1) * db_ih[i];
        m_b_hh[i] = beta1 * m_b_hh[i] + (1 - beta1) * db_hh[i];

        // Update biased second raw moment estimate
        v_b_ih[i] = beta2 * v_b_ih[i] + (1 - beta2) * db_ih[i] * db_ih[i];
        v_b_hh[i] = beta2 * v_b_hh[i] + (1 - beta2) * db_hh[i] * db_hh[i];

        // Compute bias-corrected first moment estimate
        float m_hat_b_ih = m_b_ih[i] / (1 - std::pow(beta1, t));
        float m_hat_b_hh = m_b_hh[i] / (1 - std::pow(beta1, t));

        // Compute bias-corrected second raw moment estimate
        float v_hat_b_ih = v_b_ih[i] / (1 - std::pow(beta2, t));
        float v_hat_b_hh = v_b_hh[i] / (1 - std::pow(beta2, t));

        // Update parameters
        bias_ih[i] -= alpha_t * m_hat_b_ih / (std::sqrt(v_hat_b_ih) + epsilon);
        bias_hh[i] -= alpha_t * m_hat_b_hh / (std::sqrt(v_hat_b_hh) + epsilon);
    }

    // Set updated weights and biases
    generator.set_weight_ih_input(weight_ih);
    generator.set_weight_hh_input(weight_hh);
    generator.set_bias_ih_input(bias_ih);
    generator.set_bias_hh_input(bias_hh);
}

void AnomalousThresholdGenerator::train(int num_epochs, float learning_rate, const std::vector<float>& data_to_learn) {
    // Prepare data using sliding windows
    auto [x, y] = sliding_windows(data_to_learn, lookback_len, prediction_len);

    // Initialize Adam optimizer parameters
    init_adam_parameters();

    std::vector<float> loss_history;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        generator.train(); 
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < x.size(); i += config::batch_size) {
            // Zero the gradients
            generator.zero_grad();

            float batch_loss = 0.0f;
            std::vector<float> batch_doutput;

            // Process a batch
            for (size_t j = i; j < std::min(i + static_cast<size_t>(config::batch_size), x.size()); ++j) {
                // Forward pass
                auto output = generator.forward(x[j]);

                // Compute loss
                const float epsilon = 1e-8f;
                float sample_loss = 0.0f;
                for (size_t k = 0; k < output.size(); ++k) {
                    float error = output[k] - y[j][k];
                    sample_loss += error * error;
                }
                sample_loss = sample_loss / (output.size() + epsilon);
                batch_loss += sample_loss;

                // Prepare gradients for backward pass
                std::vector<float> sample_doutput(output.size(), 2.0f / output.size());
                for (size_t k = 0; k < output.size(); ++k) {
                    sample_doutput[k] *= (output[k] - y[j][k]);
                }
                batch_doutput.insert(batch_doutput.end(), sample_doutput.begin(), sample_doutput.end());
            }

            batch_loss /= std::min(static_cast<size_t>(config::batch_size), x.size() - i);
            epoch_loss += batch_loss;

            // Backward pass
            auto [dh, dc, dw_ih, dw_hh, db_ih, db_hh] = backward_step(x[i], h, c, batch_doutput);

            // Update parameters using Adam
            update_parameters_adam(dw_ih, dw_hh, db_ih, db_hh, learning_rate);
        }

        epoch_loss /= (x.size() / config::batch_size);
        loss_history.push_back(epoch_loss);

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << ", Loss: " << epoch_loss << std::endl;

        // Early stopping
        if (loss_history.size() > config::patience) {
            bool should_stop = true;
            for (int i = 1; i <= config::patience; ++i) {
                if (loss_history[loss_history.size() - i] < loss_history[loss_history.size() - i - 1] - config::min_delta) {
                    should_stop = false;
                    break;
                }
            }
            if (should_stop) {
                std::cout << "Early stopping triggered. No improvement for " << config::patience << " epochs." << std::endl;
                break;
            }
        }
    }
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
AnomalousThresholdGenerator::sliding_windows(const std::vector<float>& data, int window_size, int prediction_len) {
    std::vector<std::vector<float>> x, y;
    for (size_t i = window_size; i < data.size(); ++i) {
        x.push_back(std::vector<float>(data.begin() + i - window_size, data.begin() + i));
        y.push_back(std::vector<float>(data.begin() + i, std::min(data.begin() + i + prediction_len, data.end())));
    }
    return {x, y};
}