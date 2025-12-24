
# SecurityNeuralNetwork
Development of an Artificial Neural Network for Detecting and Classifying Anomalies in the Computer System Functioning Process
=======
# Windows Telemetry Anomaly Detection (MLP from Scratch)

This repository contains a small, self-contained machine learning pipeline for **binary anomaly detection** and **multiclass attack-type classification** on Windows telemetry exported to CSV.

The focus of the project is an educational, “from-scratch” implementation of a **fully-connected multilayer perceptron (MLP)** with optional **GPU acceleration (CuPy)**, plus a reproducible data-preprocessing workflow and a simple architecture search script.

---

## What’s inside

### 1) Data pipeline (`data_utils.py`)
Given a CSV file, the pipeline:

- Expects two target columns:
  - `label` — binary label (0 = normal, 1 = anomaly)
  - `type` — string class name for the multiclass task (attack category)
- Converts all remaining columns to numeric features:
  - non-numeric values are coerced to `NaN`
  - missing values are filled with per-feature medians
  - drops columns that are completely empty (all `NaN`)
  - drops constant columns (no variance)
- Splits data into **train/validation** with a fixed `random_state`
- Applies optional scaling:
  - `zscore` or `minmax`
- Applies optional oversampling:
  - binary oversampling for the detector
  - multiclass oversampling (via class replication) for the classifier
- Optionally saves prepared datasets to `.npz`

### 2) Neural network (`network.py`)
A fully-connected MLP that supports:

- Weight initialization: **Xavier (uniform/normal)**, **He**, or uniform fallback
- Activations: **ReLU**, **tanh**, **sigmoid**, **GELU**, **softmax**
- Losses: **BCE**, **Cross-Entropy**, **MSE**
- Optimizer: **Adam** (plus simple SGD fallback)
- Optional **Layer Normalization** on hidden layers
- Early stopping (by validation loss)
- Saving/loading the model to/from `.npz`
- Runtime backend switch:
  - CPU (NumPy) by default
  - GPU (CuPy) if installed and enabled

### 3) Metrics (`metrics.py`)
Lightweight multiclass metrics:

- `accuracy_mc`
- `macro_f1_mc`
- `confusion_matrix_mc`

### 4) Training scripts
- `main.py` — trains a **detector** (binary) and **classifier** (multiclass) for a single configuration.
- `arch_search.py` — runs an **architecture search over the number of hidden layers** and logs:
  - train/val loss, accuracy, F1 (detector), macro-F1 (classifier)
  - training time and inference speed (samples/sec)
  - summary CSV + plots

---

## Project structure

```
.
├── main.py
├── arch_search.py
├── data_utils.py
├── network.py
├── metrics.py
├── requirements.txt
├── models/                # saved model weights (.npz) if enabled
├── logs/                  # training logs if enabled
└── plots/                 # saved plots if enabled
```

Folders are created automatically depending on the config flags.

---

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Optional GPU (CuPy)
GPU support is **optional**. If you want to run on GPU, install CuPy for your CUDA version (the exact package depends on your CUDA toolkit and OS), then set `USE_GPU = True` in `main.py` / `arch_search.py`.

If CuPy is not installed (or fails to import), the code automatically falls back to NumPy (CPU).

---

## Dataset format

Your CSV must contain:

- `label` (0/1)
- `type` (class name, e.g., `"normal"`, `"ddos"`, etc.)

All other columns are treated as candidate features.

Example (minimal):

| f1 | f2 | ... | label | type |
|---:|---:|-----|------:|------|
| 0.1 | 4.2 | ... | 0 | normal |
| 0.8 | 1.3 | ... | 1 | ddos |

---

## Quickstart

1) Put your CSV in the project root (or update `CSV_PATH` in scripts).
2) Run training:

```bash
python main.py
```

3) Run architecture search:

```bash
python arch_search.py
```

---

## Configuration

All configuration is stored as uppercase constants at the top of each script:

- Scaling:
  - `SCALING_DETECTOR` / `SCALING_CLASSIFIER` (`"zscore"` or `"minmax"`)
- Oversampling:
  - `OVERSAMPLE_DETECTOR`, `OVERSAMPLE_CLASSIFIER`
- Model depth:
  - `DET_HIDDEN_LAYERS`, `CLS_HIDDEN_LAYERS` (or `HIDDEN_LAYERS_LIST` in `arch_search.py`)
- Activations and losses:
  - `DET_HIDDEN_ACTIVATION`, `DET_OUTPUT_ACTIVATION`, `DET_LOSS`
  - `CLS_HIDDEN_ACTIVATION`, `CLS_OUTPUT_ACTIVATION`, `CLS_LOSS`
- Training:
  - learning rate, batch size, epochs, early stopping

---

## Outputs

Depending on flags in `main.py` / `arch_search.py`, the project can produce:

- `models/*.npz` — saved network parameters
- `logs/*.txt` — detailed logs per run (config + epoch history + final metrics)
- `logs/arch_search/*.csv` — summary of architecture search results
- `plots/*.png` and `plots/arch_search/*.png` — training curves and summary plots

---

## Notes

- The implementation is intentionally lightweight and educational (no scikit-learn dependency).
- Feature handling is robust to mixed types, but the quality of results depends on the CSV data and the signal in the features.
- For reproducibility, control randomness via `RANDOM_STATE` where available.


