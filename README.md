An implementation of Adaptive Anomaly Detector (AdapAD) for embedded systems with limited computational resources. AdapAD is an prediction-based AD model using LSTM for automatic data quality control.

The model utilizes online learning to adapt to concept-drift and to detect anomalous measurements from real-time univariate time series data.


## References
> Nguyen, N.T., Heldal, R. and Pelliccione, P., 2024. Concept-drift-adaptive anomaly detector for marine sensor data streams. Internet of Things, p.101414.

```bibtex
@article{nguyen2024concept,
  title={Concept-drift-adaptive anomaly detector for marine sensor data streams},
  author={Nguyen, Ngoc-Thanh and Heldal, Rogardt and Pelliccione, Patrizio},
  journal={Internet of Things},
  pages={101414},
  year={2024},
  publisher={Elsevier}
}
```
https://github.com/ntnguyen-so/AdapAD_alg

## Installation

Make sure makefile suits your CPU architecture and Operating System. Config path is specified in Main.cpp, other paths/hyperparameters are defined in Config.yaml. 

If you use included Makefile:

Clean
```
make clean
```
Build
```
make
```
Run
```
./adapad
```

## Performance on ARM Cortex-A7 528 MHz

Training time:           2.55s

Decision-making time: 0.028 Seconds per time step. (Online learning stage - this includes prediction/forward pass and complete backpropagation)

## Memory usage

2 MB - Measured with heaptrack on host system (ubuntu)


## Model comparsion - Tide pressure validation dataset
![Screenshot 2025-03-10 at 15 01 06](https://github.com/user-attachments/assets/ff04a6e6-28ca-4393-af57-b29c016c7a55)
