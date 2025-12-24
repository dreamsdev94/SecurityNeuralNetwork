import os
import time
from datetime import datetime

import numpy as np

from data_utils import create_dataset
from network import Network, set_backend
from metrics import accuracy_mc, macro_f1_mc

# ============================================================
# CONFIG — ALL SETTINGS IN ONE PLACE
# ============================================================

# --- General ---
USE_GPU = False

CSV_PATH = "Train_Test_Windows_10_plus_MyVM.csv"

# Normalization
SCALING_DETECTOR = "minmax"      # "zscore" or "minmax"
SCALING_CLASSIFIER = "zscore"

# Oversampling
OVERSAMPLE_DETECTOR = True
OVERSAMPLE_CLASSIFIER = True

# Split and randomness
VAL_RATIO = 0.2
RANDOM_STATE = 42

# --- Architecture search ---
# list of hidden-layer counts to test
HIDDEN_LAYERS_LIST = [1, 2, 3, 4, 5]

# starting size of the first hidden layer
DET_START_SIZE = 128
CLS_START_SIZE = 128

# minimum hidden layer size
MIN_HIDDEN_SIZE = 16

# weight initialization
DET_INIT = "xavier_normal"       # "xavier_normal", "xavier_uniform", "he"
CLS_INIT = "xavier_normal"

# --- Activations and loss ---
DET_HIDDEN_ACTIVATION = "sigmoid"       # "relu", "gelu", "tanh", "sigmoid"
CLS_HIDDEN_ACTIVATION = "tanh"          # "relu", "gelu", "tanh", "sigmoid"

DET_OUTPUT_ACTIVATION = "sigmoid"
CLS_OUTPUT_ACTIVATION = "softmax"

DET_LOSS = "bce"                 # "bce" or "mse"
CLS_LOSS = "ce"                  # "ce" or "mse"

# --- LayerNorm ---
USE_LAYER_NORM = False
LAYER_NORM_EVERY_K = 0           # 1 = every layer, 2 = every second, <=0 = disable LN

# --- Training ---
DET_LR = 1e-3
CLS_LR = 1e-3

DET_BATCH_SIZE = 128
CLS_BATCH_SIZE = 128

DET_MAX_EPOCHS = 50
CLS_MAX_EPOCHS = 50

EARLY_STOPPING = True
DET_PATIENCE = 5
CLS_PATIENCE = 5
MIN_DELTA = 1e-4

# --- Saving / logging / plots ---
SAVE_LOGS = True                 # save the summary CSV for architectures
SAVE_PLOTS = True                # build and save summary plots
DEBUG_TRAINING_OUTPUT = True     # show epochs in fit(debug_show=...)

# separate plot toggles
PLOT_DETECTOR = True             # if False — detector plots are not generated
PLOT_CLASSIFIER = True           # if False — classifier plots are not generated

LOGS_DIR = "logs"
PLOTS_DIR = "plots"

# dedicated folders for architecture search (inside LOGS_DIR and PLOTS_DIR)
ARCH_LOGS_DIR = os.path.join(LOGS_DIR, "arch_search")
ARCH_PLOTS_DIR = os.path.join(PLOTS_DIR, "arch_search")

# ============================================================
# Helper functions
# ============================================================


def ensure_dirs():
    """
    Creates (if they do not exist) directories for:
      - general logs (LOGS_DIR),
      - general plots (PLOTS_DIR),
      - architecture-search logs (ARCH_LOGS_DIR),
      - architecture-search plots (ARCH_PLOTS_DIR).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ARCH_LOGS_DIR, exist_ok=True)
    os.makedirs(ARCH_PLOTS_DIR, exist_ok=True)


def build_deep_architecture(input_dim: int,
                            n_hidden: int,
                            start_size: int,
                            min_hidden: int,
                            output_dim: int):
    """
    Builds a list of layer sizes:
      [input_dim, h1, h2, ... hN, output_dim]

    Each next hidden layer is reduced by ~x0.75,
    but not below min_hidden.
    This yields a "pyramidal" architecture.
    """
    hidden_sizes = []
    cur = start_size
    for _ in range(n_hidden):
        hidden_sizes.append(cur)
        # next layer is 75% of the previous, but not below min_hidden
        next_size = int(cur * 0.75)
        if next_size < min_hidden:
            next_size = min_hidden
        cur = next_size

    return [input_dim] + hidden_sizes + [output_dim]


def get_config_dict():
    """
    Collects all global ALL-CAPS parameters into a dict,
    to pass into fit(..., config_dict=...) and keep a complete
    snapshot of settings for each architecture-search run.
    """
    import sys
    module = sys.modules[__name__]
    cfg = {}
    for name, value in module.__dict__.items():
        # take all UPPERCASE variables (config settings)
        if name.isupper() and not name.startswith("_"):
            cfg[name] = value
    return cfg


def measure_inference_speed(net: Network,
                            X: np.ndarray,
                            hidden_activation: str,
                            output_activation: str,
                            n_runs: int = 5):
    """
    Measures inference throughput: samples/sec processed by the network on validation.

    n_runs — number of full passes through X using net.predict
             to get a more stable time estimate.
    """
    if X.shape[0] == 0:
        # if the validation set is empty — return 0
        return 0.0, 0.0

    total_samples = 0
    t0 = time.time()
    for _ in range(n_runs):
        _ = net.predict(
            X,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        total_samples += X.shape[0]
    elapsed = time.time() - t0
    if elapsed <= 0:
        # division-by-zero guard — if elapsed is extremely small/zero
        return float("inf"), 0.0
    # samples/sec, total elapsed time
    return total_samples / elapsed, elapsed


def compute_detector_metrics(y_true: np.ndarray,
                             y_pred_proba: np.ndarray):
    """
    Returns accuracy, precision, recall, f1 and TN/FP/FN/TP
    for a binary detector (anomaly / normal).

    y_true       — true labels (0/1) of shape (N,1) or (N,),
    y_pred_proba — probabilities (sigmoid output) of shape (N,1) or (N,).
    """
    # flatten labels and probabilities
    y_true = y_true.reshape(-1).astype(int)
    y_pred_proba = y_pred_proba.reshape(-1)
    # threshold 0.5 -> binary prediction
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # confusion-matrix components
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))

    # basic binary metrics
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def compute_classifier_metrics(y_true: np.ndarray,
                               y_pred_proba: np.ndarray):
    """
    Accuracy and Macro-F1 for a multiclass classifier.

    y_true       — one-hot matrix (N, C),
    y_pred_proba — probabilities / logits (N, C).
    """
    # NOTE: accuracy_mc / macro_f1_mc expect (Y_pred, Y_true).
    # This order matters for Macro-F1 (not symmetric).
    acc = accuracy_mc(y_pred_proba, y_true)
    macro_f1 = macro_f1_mc(y_pred_proba, y_true)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def plot_metric_vs_layers(layers_list, series_dict, title, ylabel, filename):
    """
    Plots metric(s) vs number of hidden layers.

    layers_list: list of n_hidden values (number of hidden layers),
    series_dict: {label: [v1, v2, ...]} — metric series per label,
    title/ylabel: plot labels,
    filename: output PNG name inside ARCH_PLOTS_DIR.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    for label, values in series_dict.items():
        # draw a curve for each series
        plt.plot(layers_list, values, marker="o", label=label)
    plt.xlabel("Number of hidden layers")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if len(series_dict) > 1:
        plt.legend()
    path = os.path.join(ARCH_PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[PLOT] {path}")


# ============================================================
# Main architecture-search logic
# ============================================================

def main():
    # Create all required directories for logs and plots
    ensure_dirs()
    # Select backend (CPU / GPU)
    set_backend(USE_GPU)

    print("=== CREATING DATASETS FOR ARCH SEARCH ===")
    # Create separate datasets for detector and classifier
    det_ds, cls_ds = create_dataset(
        csv_path=CSV_PATH,
        scaling_detector=SCALING_DETECTOR,
        scaling_classifier=SCALING_CLASSIFIER,
        oversample_detector=OVERSAMPLE_DETECTOR,
        oversample_classifier=OVERSAMPLE_CLASSIFIER,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
        save_npz=False,  # do not save .npz in architecture search
    )

    # Full config snapshot for logging in fit()
    config_dict = get_config_dict()
    # Accumulate results for each n_hidden here
    summary_rows = []

    # Input/output sizes remain fixed; only the number of hidden layers changes
    in_det = det_ds.X_train.shape[1]
    in_cls = cls_ds.X_train.shape[1]
    out_det = 1
    out_cls = cls_ds.y_train.shape[1]

    # Try different hidden-layer counts
    for n_hidden in HIDDEN_LAYERS_LIST:
        print("\n" + "=" * 60)
        print(f"  TESTING NUMBER OF HIDDEN LAYERS: {n_hidden}")
        print("=" * 60)

        # --- build architectures for detector and classifier ---
        det_arch = build_deep_architecture(
            input_dim=in_det,
            n_hidden=n_hidden,
            start_size=DET_START_SIZE,
            min_hidden=MIN_HIDDEN_SIZE,
            output_dim=out_det,
        )
        cls_arch = build_deep_architecture(
            input_dim=in_cls,
            n_hidden=n_hidden,
            start_size=CLS_START_SIZE,
            min_hidden=MIN_HIDDEN_SIZE,
            output_dim=out_cls,
        )

        print("Detector architecture:", det_arch)
        print("Classifier architecture:", cls_arch)

        # ======================================================
        # DETECTOR
        # ======================================================
        # Create a detector network for the current n_hidden
        det_net = Network(
            det_arch,
            init=DET_INIT,
            use_layernorm=USE_LAYER_NORM,
            ln_every_k=LAYER_NORM_EVERY_K,
        )

        # Train detector + measure training time
        t0 = time.time()
        det_history = det_net.fit(
            det_ds.X_train,
            det_ds.y_train,
            det_ds.X_val,
            det_ds.y_val,
            hidden_activation=DET_HIDDEN_ACTIVATION,
            output_activation=DET_OUTPUT_ACTIVATION,
            loss=DET_LOSS,
            optimizer="adam",
            lr=DET_LR,
            batch_size=DET_BATCH_SIZE,
            max_epochs=DET_MAX_EPOCHS,
            early_stopping=EARLY_STOPPING,
            patience=DET_PATIENCE,
            min_delta=MIN_DELTA,
            debug_show=DEBUG_TRAINING_OUTPUT,
            save_model=False,              # do not save weights in arch search
            model_type=f"detector_L{n_hidden}",
            config_dict=config_dict,
            log_dir=None,                  # do not write per-epoch logs here
            plot_metrics=False,            # disable per-epoch plots
            plots_dir=None,
        )
        det_train_time = time.time() - t0
        # last epoch index (last value in history["epoch"])
        det_epochs = int(det_history["epoch"][-1]) if det_history.get("epoch") else 0

        # final train/val loss (last epoch)
        det_train_loss_final = np.nan
        det_val_loss_final = np.nan
        if det_history.get("train_loss"):
            det_train_loss_final = float(det_history["train_loss"][-1])
        if det_history.get("val_loss"):
            det_val_loss_final = float(det_history["val_loss"][-1])

        # predictions for train/val (probabilities)
        y_pred_det_proba_val = det_net.predict(
            det_ds.X_val,
            hidden_activation=DET_HIDDEN_ACTIVATION,
            output_activation=DET_OUTPUT_ACTIVATION,
        )
        y_pred_det_proba_train = det_net.predict(
            det_ds.X_train,
            hidden_activation=DET_HIDDEN_ACTIVATION,
            output_activation=DET_OUTPUT_ACTIVATION,
        )

        # Binary metrics for train/val
        det_metrics_val = compute_detector_metrics(det_ds.y_val, y_pred_det_proba_val)
        det_metrics_train = compute_detector_metrics(det_ds.y_train, y_pred_det_proba_train)

        # Inference speed on validation
        det_speed, _ = measure_inference_speed(
            det_net,
            det_ds.X_val,
            hidden_activation=DET_HIDDEN_ACTIVATION,
            output_activation=DET_OUTPUT_ACTIVATION,
            n_runs=5,
        )

        # ======================================================
        # CLASSIFIER
        # ======================================================
        # Create a classifier network for the current n_hidden
        cls_net = Network(
            cls_arch,
            init=CLS_INIT,
            use_layernorm=USE_LAYER_NORM,
            ln_every_k=LAYER_NORM_EVERY_K,
        )

        # Train classifier + measure training time
        t0 = time.time()
        cls_history = cls_net.fit(
            cls_ds.X_train,
            cls_ds.y_train,
            cls_ds.X_val,
            cls_ds.y_val,
            hidden_activation=CLS_HIDDEN_ACTIVATION,
            output_activation=CLS_OUTPUT_ACTIVATION,
            loss=CLS_LOSS,
            optimizer="adam",
            lr=CLS_LR,
            batch_size=CLS_BATCH_SIZE,
            max_epochs=CLS_MAX_EPOCHS,
            early_stopping=EARLY_STOPPING,
            patience=CLS_PATIENCE,
            min_delta=MIN_DELTA,
            debug_show=DEBUG_TRAINING_OUTPUT,
            save_model=False,
            model_type=f"classifier_L{n_hidden}",
            config_dict=config_dict,
            log_dir=None,
            plot_metrics=False,
            plots_dir=None,
        )
        cls_train_time = time.time() - t0
        cls_epochs = int(cls_history["epoch"][-1]) if cls_history.get("epoch") else 0

        cls_train_loss_final = np.nan
        cls_val_loss_final = np.nan
        if cls_history.get("train_loss"):
            cls_train_loss_final = float(cls_history["train_loss"][-1])
        if cls_history.get("val_loss"):
            cls_val_loss_final = float(cls_history["val_loss"][-1])

        # Predictions (one-hot probabilities) for train/val
        y_pred_cls_proba_val = cls_net.predict(
            cls_ds.X_val,
            hidden_activation=CLS_HIDDEN_ACTIVATION,
            output_activation=CLS_OUTPUT_ACTIVATION,
        )
        y_pred_cls_proba_train = cls_net.predict(
            cls_ds.X_train,
            hidden_activation=CLS_HIDDEN_ACTIVATION,
            output_activation=CLS_OUTPUT_ACTIVATION,
        )

        # Multiclass metrics (accuracy + macro-F1) for train/val
        cls_metrics_val = compute_classifier_metrics(cls_ds.y_val, y_pred_cls_proba_val)
        cls_metrics_train = compute_classifier_metrics(cls_ds.y_train, y_pred_cls_proba_train)

        # Inference speed on validation
        cls_speed, _ = measure_inference_speed(
            cls_net,
            cls_ds.X_val,
            hidden_activation=CLS_HIDDEN_ACTIVATION,
            output_activation=CLS_OUTPUT_ACTIVATION,
            n_runs=5,
        )

        # --- final validation errors (1 - F1 / 1 - macro-F1) ---
        det_error_val = 1.0 - float(det_metrics_val["f1"])
        cls_error_val = 1.0 - float(cls_metrics_val["macro_f1"])

        # Store all key numbers for the current n_hidden
        summary_rows.append({
            "hidden_layers": n_hidden,

            # DETECTOR
            "det_train_loss": det_train_loss_final,
            "det_val_loss": det_val_loss_final,
            "det_train_acc": det_metrics_train["accuracy"],
            "det_train_f1": det_metrics_train["f1"],
            "det_val_acc": det_metrics_val["accuracy"],
            "det_val_f1": det_metrics_val["f1"],
            "det_error_1_minus_f1": det_error_val,
            "det_train_time_sec": det_train_time,
            "det_epochs": det_epochs,
            "det_speed_samples_per_sec": det_speed,

            # CLASSIFIER
            "cls_train_loss": cls_train_loss_final,
            "cls_val_loss": cls_val_loss_final,
            "cls_train_acc": cls_metrics_train["accuracy"],
            "cls_train_macro_f1": cls_metrics_train["macro_f1"],
            "cls_val_acc": cls_metrics_val["accuracy"],
            "cls_val_macro_f1": cls_metrics_val["macro_f1"],
            "cls_error_1_minus_macro_f1": cls_error_val,
            "cls_train_time_sec": cls_train_time,
            "cls_epochs": cls_epochs,
            "cls_speed_samples_per_sec": cls_speed,
        })

    # ============================================================
    # Save the summary CSV log
    # ============================================================
    if SAVE_LOGS:
        # timestamp in filename to avoid overwriting previous logs
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(ARCH_LOGS_DIR, f"arch_search_layers_{ts}.csv")

        import csv
        # column order in the summary file
        fieldnames = [
            "hidden_layers",

            "det_train_loss", "det_val_loss",
            "det_train_acc", "det_train_f1",
            "det_val_acc", "det_val_f1",
            "det_error_1_minus_f1",
            "det_train_time_sec", "det_epochs", "det_speed_samples_per_sec",

            "cls_train_loss", "cls_val_loss",
            "cls_train_acc", "cls_train_macro_f1",
            "cls_val_acc", "cls_val_macro_f1",
            "cls_error_1_minus_macro_f1",
            "cls_train_time_sec", "cls_epochs", "cls_speed_samples_per_sec",
        ]
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            # delimiter ";" is convenient for Excel/LibreOffice
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        print(f"\n[LOG] Summary CSV saved to {log_path}")
    else:
        print("\n[LOG] SAVE_LOGS = False, CSV log was not saved.")

    # ============================================================
    # Build summary plots
    # ============================================================
    if SAVE_PLOTS:
        # list of n_hidden values tested
        layers = [row["hidden_layers"] for row in summary_rows]

        # Detector series
        det_train_loss_vals = [row["det_train_loss"] for row in summary_rows]
        det_val_loss_vals = [row["det_val_loss"] for row in summary_rows]
        det_train_acc_vals = [row["det_train_acc"] for row in summary_rows]
        det_val_acc_vals = [row["det_val_acc"] for row in summary_rows]
        det_train_f1_vals = [row["det_train_f1"] for row in summary_rows]
        det_val_f1_vals = [row["det_val_f1"] for row in summary_rows]
        det_error_vals = [row["det_error_1_minus_f1"] for row in summary_rows]
        det_time_vals = [row["det_train_time_sec"] for row in summary_rows]
        det_epochs_vals = [row["det_epochs"] for row in summary_rows]
        det_speed_vals = [row["det_speed_samples_per_sec"] for row in summary_rows]

        # Classifier series
        cls_train_loss_vals = [row["cls_train_loss"] for row in summary_rows]
        cls_val_loss_vals = [row["cls_val_loss"] for row in summary_rows]
        cls_train_acc_vals = [row["cls_train_acc"] for row in summary_rows]
        cls_val_acc_vals = [row["cls_val_acc"] for row in summary_rows]
        cls_train_macro_f1_vals = [row["cls_train_macro_f1"] for row in summary_rows]
        cls_val_macro_f1_vals = [row["cls_val_macro_f1"] for row in summary_rows]
        cls_error_vals = [row["cls_error_1_minus_macro_f1"] for row in summary_rows]
        cls_time_vals = [row["cls_train_time_sec"] for row in summary_rows]
        cls_epochs_vals = [row["cls_epochs"] for row in summary_rows]
        cls_speed_vals = [row["cls_speed_samples_per_sec"] for row in summary_rows]

        if PLOT_DETECTOR:
            # Loss train/val
            plot_metric_vs_layers(
                layers,
                {"train_loss": det_train_loss_vals, "val_loss": det_val_loss_vals},
                "DETECTOR: loss (train/val) vs number of layers",
                "Loss",
                "detector_loss_vs_layers.png",
            )
            # F1 train/val
            plot_metric_vs_layers(
                layers,
                {"train_F1": det_train_f1_vals, "val_F1": det_val_f1_vals},
                "DETECTOR: F1 (train/val) vs number of layers",
                "F1",
                "detector_f1_vs_layers.png",
            )
            # Accuracy train/val
            plot_metric_vs_layers(
                layers,
                {"train_acc": det_train_acc_vals, "val_acc": det_val_acc_vals},
                "DETECTOR: accuracy (train/val) vs number of layers",
                "Accuracy",
                "detector_acc_vs_layers.png",
            )
            # Training progress: epochs
            plot_metric_vs_layers(
                layers,
                {"epochs": det_epochs_vals},
                "DETECTOR: number of epochs vs number of layers",
                "Epochs",
                "detector_epochs_vs_layers.png",
            )
            # Training time
            plot_metric_vs_layers(
                layers,
                {"train_time_sec": det_time_vals},
                "DETECTOR: training time vs number of layers",
                "Training time, sec",
                "detector_time_vs_layers.png",
            )
            # Recognition speed
            plot_metric_vs_layers(
                layers,
                {"inference_speed": det_speed_vals},
                "DETECTOR: inference speed vs number of layers",
                "Samples/sec",
                "detector_speed_vs_layers.png",
            )
            # Error 1 - F1 (val)
            plot_metric_vs_layers(
                layers,
                {"val_error_1-F1": det_error_vals},
                "DETECTOR: error (1 - F1) vs number of layers",
                "1 - F1 (val)",
                "detector_error_vs_layers.png",
            )

        if PLOT_CLASSIFIER:
            # Loss train/val
            plot_metric_vs_layers(
                layers,
                {"train_loss": cls_train_loss_vals, "val_loss": cls_val_loss_vals},
                "CLASSIFIER: loss (train/val) vs number of layers",
                "Loss",
                "classifier_loss_vs_layers.png",
            )
            # Macro-F1 train/val
            plot_metric_vs_layers(
                layers,
                {
                    "train_macro_F1": cls_train_macro_f1_vals,
                    "val_macro_F1": cls_val_macro_f1_vals,
                },
                "CLASSIFIER: macro-F1 (train/val) vs number of layers",
                "macro-F1",
                "classifier_f1_vs_layers.png",
            )
            # Accuracy train/val
            plot_metric_vs_layers(
                layers,
                {"train_acc": cls_train_acc_vals, "val_acc": cls_val_acc_vals},
                "CLASSIFIER: accuracy (train/val) vs number of layers",
                "Accuracy",
                "classifier_acc_vs_layers.png",
            )
            # Training progress: epochs
            plot_metric_vs_layers(
                layers,
                {"epochs": cls_epochs_vals},
                "CLASSIFIER: number of epochs vs number of layers",
                "Epochs",
                "classifier_epochs_vs_layers.png",
            )
            # Training time
            plot_metric_vs_layers(
                layers,
                {"train_time_sec": cls_time_vals},
                "CLASSIFIER: training time vs number of layers",
                "Training time, sec",
                "classifier_time_vs_layers.png",
            )
            # Recognition speed
            plot_metric_vs_layers(
                layers,
                {"inference_speed": cls_speed_vals},
                "CLASSIFIER: inference speed vs number of layers",
                "Samples/sec",
                "classifier_speed_vs_layers.png",
            )
            # Error 1 - macro-F1 (val)
            plot_metric_vs_layers(
                layers,
                {"val_error_1-macroF1": cls_error_vals},
                "CLASSIFIER: error (1 - macro-F1) vs number of layers",
                "1 - macro-F1 (val)",
                "classifier_error_vs_layers.png",
            )

        print(f"[PLOTS] Plots saved to {ARCH_PLOTS_DIR}")
    else:
        print("[PLOTS] SAVE_PLOTS = False, plots were not generated.")


if __name__ == "__main__":
    # Run architecture search only when the file is executed directly
    main()
