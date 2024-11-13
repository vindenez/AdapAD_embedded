#ifndef NORMAL_DATA_PREDICTOR_HPP
#define NORMAL_DATA_PREDICTOR_HPP

#include "lstm_predictor.hpp"
#include <vector>
#include <string>

class NormalDataPredictor {
public:
    NormalDataPredictor(int lstm_layer, int lstm_unit, int lookback_len, int prediction_len);

    using EarlyStoppingCallback = std::function<bool(int epoch, float loss)>;

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    train(int num_epochs, float learning_rate, const std::vector<float>& data_to_learn,
        const EarlyStoppingCallback& callback = nullptr);    float predict(const std::vector<float>& observed);
    void update(int epoch_update, float lr_update, const std::vector<float>& past_observations, float recent_observation);

private:
    int num_layers;
    int hidden_size;
    int lookback_len;
    int prediction_len;
    LSTMPredictor predictor;
};

#endif // NORMAL_DATA_PREDICTOR_HPP
