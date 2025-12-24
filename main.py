import os
from datetime import datetime

import numpy as np

from data_utils import create_dataset
from network import Network, set_backend
from metrics import accuracy_mc, macro_f1_mc


# SETTINGS IN ONE PLACE

# --- General ---
USE_GPU = False  # whether to try using GPU (CuPy)

CSV_PATH = "Train_Test_Windows_10.csv"

# Normalization
SCALING_DETECTOR = "minmax"      # "zscore" or "minmax"
SCALING_CLASSIFIER = "zscore"

# Oversampling
OVERSAMPLE_DETECTOR = True       # balance classes for the detector
OVERSAMPLE_CLASSIFIER = True     # balance classes for the classifier

# Architecture
# number of hidden layers
DET_HIDDEN_LAYERS = 5
CLS_HIDDEN_LAYERS = 5

# starting size of the first hidden layer
DET_START_SIZE = 128
CLS_START_SIZE = 128

# minimum hidden layer size
MIN_HIDDEN_SIZE = 16

# weight initialization
DET_INIT = "xavier_normal"       # "xavier_normal", "xavier_uniform", "he"
CLS_INIT = "xavier_normal"

# Training optimizer
OPTIMIZER = None

# --- Activations and loss ---
DET_HIDDEN_ACTIVATION = "relu"       # "relu", "gelu", "tanh", "sigmoid"
CLS_HIDDEN_ACTIVATION = "tanh"       # "relu", "gelu", "tanh", "sigmoid"

DET_OUTPUT_ACTIVATION = "sigmoid"    # binary output (anomaly probability)
CLS_OUTPUT_ACTIVATION = "softmax"    # multiclass output (class distribution)

DET_LOSS = "bce"                 # "bce" or "mse"
CLS_LOSS = "ce"                  # "ce" or "mse"

# --- LayerNorm ---
USE_LAYER_NORM = False
LAYER_NORM_EVERY_K = 0           # 1 = every layer, 2 = every other layer, <=0 = disable LN

# Learning rate
DET_LR = 1e-3
CLS_LR = 1e-3

# Batch size
DET_BATCH_SIZE = 256
CLS_BATCH_SIZE = 256

# Maximum number of epochs
DET_MAX_EPOCHS = 50
CLS_MAX_EPOCHS = 50

# Number of epochs without improvement for early stopping
DET_PATIENCE = 10
CLS_PATIENCE = 10
MIN_DELTA = 1e-5

# Early stopping switch
EARLY_STOPPING = False

# --- Saving / logging / plots ---
SAVE_MODELS = True          # save .npz weights
SAVE_LOGS = True            # save txt log
SAVE_PLOTS = True           # global: build and save plots
DEBUG_TRAINING_OUTPUT = True  # print epochs to console via fit(debug_show=...)

# separate control over plotting
PLOT_DETECTOR = True        # if False — detector plots are not generated
PLOT_CLASSIFIER = True      # if False — classifier plots are not generated

MODELS_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"


# ============================================================
# Helper functions
# ============================================================

def get_config_dict():
    """
    Collects all global CAPS settings from this module
    in order to save them into the txt log.

    Important: we explicitly filter only the constants listed in set(),
    so we do not accidentally capture anything extra (e.g., imported from other modules).
    """
    import sys
    module = sys.modules[__name__]
    cfg = {}
    for name, value in module.__dict__.items():
        if name.isupper() and not name.startswith("_"):
            # filter to avoid picking up imported CONSTANTS from other modules
            if name in {
                "USE_GPU",
                "CSV_PATH",
                "SCALING_DETECTOR",
                "SCALING_CLASSIFIER",
                "OVERSAMPLE_DETECTOR",
                "OVERSAMPLE_CLASSIFIER",
                "DET_HIDDEN_LAYERS",
                "CLS_HIDDEN_LAYERS",
                "DET_START_SIZE",
                "CLS_START_SIZE",
                "MIN_HIDDEN_SIZE",
                "DET_INIT",
                "CLS_INIT",
                "DET_HIDDEN_ACTIVATION",
                "CLS_HIDDEN_ACTIVATION",
                "DET_OUTPUT_ACTIVATION",
                "CLS_OUTPUT_ACTIVATION",
                "DET_LOSS",
                "CLS_LOSS",
                "USE_LAYER_NORM",
                "LAYER_NORM_EVERY_K",
                "DET_LR",
                "CLS_LR",
                "DET_BATCH_SIZE",
                "CLS_BATCH_SIZE",
                "DET_MAX_EPOCHS",
                "CLS_MAX_EPOCHS",
                "DET_PATIENCE",
                "CLS_PATIENCE",
                "EARLY_STOPPING",
                "SAVE_MODELS",
                "SAVE_LOGS",
                "SAVE_PLOTS",
                "DEBUG_TRAINING_OUTPUT",
                "PLOT_DETECTOR",
                "PLOT_CLASSIFIER",
                "MODELS_DIR",
                "LOGS_DIR",
                "PLOTS_DIR",
            }:
                cfg[name] = value
    return cfg


def build_deep_architecture(input_dim: int,
                            n_hidden: int,
                            start_size: int,
                            min_hidden: int,
                            output_dim: int):
    """
    Builds a list of layer sizes of the form:
      [input_dim, h1, h2, ..., hN, output_dim]

    Each subsequent hidden layer shrinks by ~x0.75,
    but not below min_hidden.

    This yields a "pyramidal" architecture: a wide first layer, then narrowing.
    """
    sizes = []
    cur = start_size
    for _ in range(n_hidden):
        sizes.append(cur)
        # decrease the number of neurons, but do not go below min_hidden
        next_size = int(cur * 0.75)
        if next_size < min_hidden:
            next_size = min_hidden
        cur = next_size
    # add input and output dimensions
    return [input_dim] + sizes + [output_dim]


def ensure_dirs():
    """
    Ensures the directories for models, logs, and plots exist.
    If the flags are disabled — directories are not created.
    """
    if SAVE_MODELS:
        os.makedirs(MODELS_DIR, exist_ok=True)
    if SAVE_LOGS:
        os.makedirs(LOGS_DIR, exist_ok=True)
    if SAVE_PLOTS and (PLOT_DETECTOR or PLOT_CLASSIFIER):
        os.makedirs(PLOTS_DIR, exist_ok=True)


def save_log_txt(log_path: str,
                 model_type: str,
                 layers,
                 history: dict,
                 final_info_lines,
                 is_binary: bool):
    """
    log_path          — path to the txt file
    model_type        — "DETECTOR" or "CLASSIFIER"
    layers            — list of layer sizes
    history           — dictionary returned by Network.fit()
    final_info_lines  — list of text lines that we print to the console
    is_binary         — True for the detector, False for the classifier

    In essence, this is a "snapshot" of all settings + training history +
    final metrics for a particular run.
    """
    cfg = get_config_dict()

    with open(log_path, "w", encoding="utf-8") as f:
        # 1) Full configuration from main.py
        f.write("=== CONFIG (main.py) ===\n")
        for k in sorted(cfg.keys()):
            f.write(f"{k} = {cfg[k]}\n")
        f.write("\n")

        # 2) Model information
        f.write(f"Model type: {model_type}\n")
        f.write(f"Layers: {layers}\n\n")

        # 3) Training history by epochs
        f.write("=== TRAINING HISTORY ===\n")
        epochs = history.get("epoch", [])
        for i, ep in enumerate(epochs):
            train_loss = history["train_loss"][i]
            val_loss = history["val_loss"][i]
            if is_binary:
                acc = history["val_acc"][i]
                prec = history["val_precision"][i]
                rec = history["val_recall"][i]
                f1 = history["val_f1"][i]
                f.write(
                    f"Epoch {ep:03d}: "
                    f"train_loss={train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"val_acc={acc:.4f}  "
                    f"val_prec={prec:.4f}  "
                    f"val_rec={rec:.4f}  "
                    f"val_f1={f1:.4f}\n"
                )
            else:
                acc = history["val_acc"][i]
                macro_f1 = history["val_macro_f1"][i]
                f.write(
                    f"Epoch {ep:03d}: "
                    f"train_loss={train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"val_acc={acc:.4f}  "
                    f"val_macroF1={macro_f1:.4f}\n"
                )

        # 4) Final metrics for the best model
        f.write("\n=== FINAL METRICS ===\n")
        for line in final_info_lines:
            f.write(line + "\n")


def plot_history(history: dict, base_name: str, out_dir: str):
    """
    Builds and saves plots from the history returned by Network.fit().
    Creates several .png files in the specified folder.

    base_name — base part of file names (without extension),
    out_dir   — directory where plots are saved.
    """
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    if not epochs:
        # if history is empty — nothing to plot
        return

    # Loss plot (train / val)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{base_name} — Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, base_name + "_loss.png"))
    plt.close()

    # Binary or multiclass variant
    if "val_macro_f1" in history:
        # multiclass classifier
        val_acc = history.get("val_acc", [])
        val_macro_f1 = history.get("val_macro_f1", [])

        plt.figure()
        plt.plot(epochs, val_acc, label="val_acc")
        plt.plot(epochs, val_macro_f1, label="val_macroF1")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title(f"{base_name} — Accuracy & Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, base_name + "_acc_macroF1.png"))
        plt.close()

    elif "val_precision" in history:
        # binary detector
        val_acc = history.get("val_acc", [])
        val_prec = history.get("val_precision", [])
        val_rec = history.get("val_recall", [])
        val_f1 = history.get("val_f1", [])

        plt.figure()
        plt.plot(epochs, val_acc, label="val_acc")
        plt.plot(epochs, val_prec, label="val_prec")
        plt.plot(epochs, val_rec, label="val_rec")
        plt.plot(epochs, val_f1, label="val_f1")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title(f"{base_name} — Accuracy / Precision / Recall / F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, base_name + "_metrics.png"))
        plt.close()


# ============================================================
# main
# ============================================================

def main():
    # 1) GPU / CPU — choose the computation backend
    set_backend(USE_GPU)

    # 2) Folders for models, logs, and plots
    ensure_dirs()

    # 3) Load and prepare the data
    # create_dataset:
    #   - reads the CSV,
    #   - cleans/transforms features,
    #   - creates a train/val split,
    #   - scales (zscore/minmax),
    #   - performs oversampling (if needed),
    #   - returns two Datasets: one for the detector and one for the classifier.
    det_ds, cls_ds = create_dataset(
        CSV_PATH,
        scaling_detector=SCALING_DETECTOR,
        scaling_classifier=SCALING_CLASSIFIER,
        oversample_detector=OVERSAMPLE_DETECTOR,
        oversample_classifier=OVERSAMPLE_CLASSIFIER,
        save_npz=True,  # also save the prepared datasets to .npz in parallel
    )

    # 4) Architectures
    # input sizes for both networks (number of features)
    in_det = det_ds.X_train.shape[1]
    in_cls = cls_ds.X_train.shape[1]
    # detector output — one neuron (anomaly probability)
    out_det = 1
    # classifier output — number of classes (one-hot)
    out_cls = cls_ds.y_train.shape[1]

    det_arch = build_deep_architecture(
        input_dim=in_det,
        n_hidden=DET_HIDDEN_LAYERS,
        start_size=DET_START_SIZE,
        min_hidden=MIN_HIDDEN_SIZE,
        output_dim=out_det,
    )
    cls_arch = build_deep_architecture(
        input_dim=in_cls,
        n_hidden=CLS_HIDDEN_LAYERS,
        start_size=CLS_START_SIZE,
        min_hidden=MIN_HIDDEN_SIZE,
        output_dim=out_cls,
    )

    # Add a timestamp so files do not overwrite each other between runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    det_base = f"detector_{DET_HIDDEN_LAYERS}layers_{timestamp}"
    cls_base = f"classifier_{CLS_HIDDEN_LAYERS}layers_{timestamp}"

    print("\n=== TRAINING DETECTOR ===")
    print(f"Detector architecture: {det_arch}")

    # 5) Detector
    # Create a network for the binary task
    det_net = Network(
        det_arch,
        init=DET_INIT,
        use_layernorm=USE_LAYER_NORM,
        ln_every_k=LAYER_NORM_EVERY_K,
    )

    # Train the detector
    det_history = det_net.fit(
        det_ds.X_train,
        det_ds.y_train,
        det_ds.X_val,
        det_ds.y_val,
        hidden_activation=DET_HIDDEN_ACTIVATION,
        output_activation=DET_OUTPUT_ACTIVATION,
        loss=DET_LOSS,
        optimizer=OPTIMIZER,
        lr=DET_LR,
        batch_size=DET_BATCH_SIZE,
        max_epochs=DET_MAX_EPOCHS,
        early_stopping=EARLY_STOPPING,
        patience=DET_PATIENCE,
        min_delta=MIN_DELTA,
        debug_show=DEBUG_TRAINING_OUTPUT,
    )

    # Prediction on validation for the detector's final metrics
    y_true_det = det_ds.y_val.reshape(-1).astype(int)
    y_pred_proba_det = det_net.predict(
        det_ds.X_val,
        hidden_activation=DET_HIDDEN_ACTIVATION,
        output_activation=DET_OUTPUT_ACTIVATION,
    ).reshape(-1)
    # Convert probabilities to binary labels using the 0.5 threshold
    y_pred_det = (y_pred_proba_det >= 0.5).astype(int)

    # Compute confusion-matrix elements:
    # TN — 0 predicted as 0, TP — 1 predicted as 1, FP, FN — errors.
    tn = int(np.sum((y_pred_det == 0) & (y_true_det == 0)))
    fp = int(np.sum((y_pred_det == 1) & (y_true_det == 0)))
    fn = int(np.sum((y_pred_det == 0) & (y_true_det == 1)))
    tp = int(np.sum((y_pred_det == 1) & (y_true_det == 1)))

    # Compute classic binary metrics
    det_acc = (tp + tn) / len(y_true_det) if len(y_true_det) > 0 else 0.0
    det_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    det_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if det_prec + det_rec > 0:
        det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec)
    else:
        det_f1 = 0.0

    print("\n=== FINAL METRICS ===")
    det_lines = [
        f"[DETECTOR] Accuracy : {det_acc:.4f}",
        f"[DETECTOR] Precision: {det_prec:.4f}",
        f"[DETECTOR] Recall   : {det_rec:.4f}",
        f"[DETECTOR] F1       : {det_f1:.4f}",
        f"[DETECTOR] Confusion matrix (TN FP FN TP): {tn} {fp} {fn} {tp}",
    ]
    for line in det_lines:
        print(line)

    # 6) Classifier
    print("\n=== TRAINING CLASSIFIER ===")
    print(f"Classifier architecture: {cls_arch}")

    # Network for the multiclass task
    cls_net = Network(
        cls_arch,
        init=CLS_INIT,
        use_layernorm=USE_LAYER_NORM,
        ln_every_k=LAYER_NORM_EVERY_K,
    )

    # Train the classifier
    cls_history = cls_net.fit(
        cls_ds.X_train,
        cls_ds.y_train,
        cls_ds.X_val,
        cls_ds.y_val,
        hidden_activation=CLS_HIDDEN_ACTIVATION,
        output_activation=CLS_OUTPUT_ACTIVATION,
        loss=CLS_LOSS,
        optimizer=OPTIMIZER,
        lr=CLS_LR,
        batch_size=CLS_BATCH_SIZE,
        max_epochs=CLS_MAX_EPOCHS,
        early_stopping=EARLY_STOPPING,
        patience=CLS_PATIENCE,
        min_delta=MIN_DELTA,
        debug_show=DEBUG_TRAINING_OUTPUT,
    )

    # Convert one-hot to class indices for true and predicted labels
    y_true_cls = np.argmax(cls_ds.y_val, axis=1)
    y_pred_proba_cls = cls_net.predict(
        cls_ds.X_val,
        hidden_activation=CLS_HIDDEN_ACTIVATION,
        output_activation=CLS_OUTPUT_ACTIVATION,
    )
    y_pred_cls = np.argmax(y_pred_proba_cls, axis=1)

    # Aggregate multiclass metrics (accuracy and macro-F1)
    cls_acc = accuracy_mc(y_pred_proba_cls, cls_ds.y_val)
    cls_macro_f1 = macro_f1_mc(y_pred_proba_cls, cls_ds.y_val)

    # Confusion matrix (rows — true classes, columns — predicted)
    n_classes = cls_ds.y_val.shape[1]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_cls, y_pred_cls):
        cm[int(t), int(p)] += 1

    cls_lines = [
        f"[CLASSIFIER] Accuracy : {cls_acc:.4f}",
        f"[CLASSIFIER] Macro-F1 : {cls_macro_f1:.4f}",
        "[CLASSIFIER] Confusion matrix (rows=true, cols=pred):",
        str(cm),
    ]
    for line in cls_lines:
        print(line)

    # 7) Save models (weights and architecture) to .npz
    if SAVE_MODELS:
        det_model_path = os.path.join(MODELS_DIR, det_base + ".npz")
        cls_model_path = os.path.join(MODELS_DIR, cls_base + ".npz")
        det_net.save(det_model_path)
        cls_net.save(cls_model_path)

    # 8) Training logs (config + history + final metrics)
    if SAVE_LOGS:
        det_log_path = os.path.join(LOGS_DIR, det_base + ".txt")
        cls_log_path = os.path.join(LOGS_DIR, cls_base + ".txt")
        save_log_txt(det_log_path, "DETECTOR", det_arch, det_history, det_lines, True)
        save_log_txt(cls_log_path, "CLASSIFIER", cls_arch, cls_history, cls_lines, False)

    # 9) Plots from training history
    if SAVE_PLOTS and PLOT_DETECTOR:
        plot_history(det_history, det_base, PLOTS_DIR)
    if SAVE_PLOTS and PLOT_CLASSIFIER:
        plot_history(cls_history, cls_base, PLOTS_DIR)


if __name__ == "__main__":
    # Entry point — run main() only if the file is executed directly,
    # not imported as a module.
    main()
