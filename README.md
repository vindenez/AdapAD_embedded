An implementation of Adaptive Anomaly Detector (AdapAD) for embedded systems with limited computational resources. AdapAD is a prediction-based AD algorithm that aims to detect anomalous measurements in real-time on univariate time series data.

The algorithm was developed for marine sensor measurements as it utilizes online learning to adapt to concept-drift.

<img width="700" alt="classdiagram" src="https://github.com/user-attachments/assets/698cb3f3-fa21-4a63-a073-6d2b282f8268" />

## Performance on ARM Cortex-A7 with use_neon enabled

Training time: 25.1s

Decision-making time: 1.9s per time step (Online learning - this includes backpropagation with an intermediate forward pass per update epoch)

## Underlying neural network

<img width="800" alt="NNArchitecture" src="https://github.com/user-attachments/assets/841eb76d-7a81-49b4-8b95-8ce8090c84ec" />

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


