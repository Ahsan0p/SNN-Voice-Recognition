# Comparative Study of Traditional Machine Learning and Spiking Neural Networks for Voice Command Recognition

This project presents a comparative analysis of a traditional machine learning model and a Spiking Neural Network (SNN) for binary voice command recognition ("yes" vs "no") using the Google Speech Commands dataset.

## Overview
We compare:

- Logistic Regression with MFCC features
- Custom Spiking Neural Network with Leaky Integrate-and-Fire (LIF) neurons

Evaluation focuses on:
- Classification performance
- Inference latency
- CPU and memory usage
- Estimated energy consumption
- Spike sparsity and neuromorphic efficiency tradeoffs

## Dataset
- Google Speech Commands (v1)
- 1000 samples total  
  - 500 "yes"
  - 500 "no"
- 80/20 train-test split

## Methodology
### Feature Extraction
- Audio normalized to 1 second (16 kHz)
- 13 MFCC features extracted using Librosa
- Time-averaged feature vectors used for classification

### Traditional Model
- Logistic Regression
- Scikit-learn implementation
- L-BFGS solver

### Spiking Neural Network
- Input layer: 13 neurons
- Hidden layer: 64 LIF neurons
- Output layer: 2 neurons
- Surrogate gradient training
- 30 timesteps rate coding

## Results Summary
| Metric | Logistic Regression | SNN |
|--------|--------------------|-----|
| Accuracy | 88.5% | 78.5% |
| Latency | 1.19 ms | 12.87 ms |
| CPU Usage | 3.8% | 56.2% |

Traditional ML outperformed SNN in software simulation, while SNNs showed theoretical efficiency advantages for neuromorphic hardware.

## Repository Structure
```bash
traditionalTrain.py      # Logistic regression training
train_snn.py             # SNN training
Test&Compere.py          # Evaluation and comparison
comparison_results.txt   # Output metrics
```

## Installation
```bash
pip install numpy librosa scikit-learn torch psutil joblib
```

## Run
Train traditional model:
```bash
python traditionalTrain.py
```

Train SNN:
```bash
python train_snn.py
```

Run comparison:
```bash
python Test&Compere.py
```

## Technologies Used
- Python
- PyTorch
- Scikit-learn
- Librosa
- Neuromorphic Computing Concepts

## Authors
- Fabeha Zahid Mahmood  
- Haseeb Ahmad Sardar  
- M. Ahsan Ali

## Future Work
- Larger vocabulary recognition
- Sparsity regularization
- Deployment on neuromorphic hardware (e.g. Intel Loihi)

