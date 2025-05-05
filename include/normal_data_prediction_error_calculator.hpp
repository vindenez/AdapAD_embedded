#ifndef NORMAL_DATA_PREDICTION_ERROR_CALCULATOR_HPP
#define NORMAL_DATA_PREDICTION_ERROR_CALCULATOR_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

class NormalDataPredictionErrorCalculator {
  public:
    static float calc_error(float predicted_val, float observed_val) {
        float diff = predicted_val - observed_val;
        return diff * diff;
    }

    static std::vector<float> calc_error(const std::vector<float> &observed_vals,
                                         const std::vector<float> &predicted_vals) {

        if (observed_vals.size() != predicted_vals.size()) {
            throw std::runtime_error("Observed and predicted values must have same length");
        }

        std::vector<float> errors;
        errors.reserve(observed_vals.size());

        for (std::size_t i = 0; i < observed_vals.size(); i++) {
            errors.push_back(calc_error(predicted_vals[i], observed_vals[i]));
        }

        return errors;
    }
};

#endif // NORMAL_DATA_PREDICTION_ERROR_CALCULATOR_HPP