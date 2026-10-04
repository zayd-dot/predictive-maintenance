# Predictive Maintenance: Turbofan Remaining Useful Life

Predict remaining useful life (RUL) from multivariate turbofan sensor sequences using a stacked LSTM and a Random Forest baseline. The project uses the NASA CMAPSS dataset and implements data preparation, model training, evaluation, and visualization.

## Pipeline

1. Load engine sensor histories and calculate RUL targets.
2. Cap training RUL at 125 cycles and select 14 sensor channels.
3. Apply MinMax normalization.
4. Create overlapping 30-cycle windows.
5. Train sequence and statistical-feature models.
6. Evaluate predictions using RMSE, MAE, R², and the NASA scoring function.

## Architecture

| Component | Configuration |
|---|---|
| Dataset | NASA CMAPSS; FD001 by default |
| Input | 30 cycles × 14 sensors |
| Sequence model | Stacked LSTM with 64 and 32 units |
| Output head | Dense 32, then one RUL prediction |
| Training | Adam, MSE loss, dropout, batch normalization |
| Callbacks | Early stopping and learning-rate reduction |
| Baseline | Random Forest, 200 trees, maximum depth 15 |
| Baseline features | Mean, standard deviation, minimum, maximum, and last value per sensor |

The repository also includes BiLSTM with temporal attention and a CNN–BiLSTM–attention architecture for further experiments.

## Results

| Metric | LSTM | Random Forest |
|---|---:|---:|
| RMSE | **14.77** | 19.46 |
| MAE | **10.80** | 14.42 |
| R² | **0.87** | 0.78 |
| NASA Score | **414.83** | 981.53 |

The reported LSTM RMSE is approximately 24% lower than the Random Forest baseline. RMSE and MAE are measured in operating cycles. Lower NASA scores indicate better performance under an asymmetric error penalty.

## Quick Start

```bash
python -m pip install -r requirements.txt
python download_data.py
python train.py --dataset FD001 --baseline
```

Custom training:

```bash
python train.py --dataset FD001 --epochs 50 --window 30 --batch-size 64
```

## Project Structure

| File | Purpose |
|---|---|
| `configs/config.py` | Sensor selection and hyperparameters |
| `src/data_preprocessing.py` | Data loading, labels, scaling, and windows |
| `src/models.py` | LSTM and Random Forest |
| `src/models_enhanced.py` | Attention-based architectures |
| `src/evaluation.py` | Metrics and plots |
| `train.py` | Training and evaluation |
| `tune.py` | Configuration-comparison experiments |

## Technologies

Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.
