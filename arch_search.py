import os
import time
from datetime import datetime

import numpy as np

from data_utils import create_dataset
from network import Network, set_backend
from metrics import accuracy_mc, macro_f1_mc

# ============================================================
# CONFIG — ВСЕ НАЛАШТУВАННЯ В ОДНОМУ МІСЦІ
# ============================================================

# --- Загальне ---
USE_GPU = False

CSV_PATH = "Train_Test_Windows_10_plus_MyVM.csv"

# Нормалізація
SCALING_DETECTOR = "minmax"      # "zscore" або "minmax"
SCALING_CLASSIFIER = "zscore"

# Oversampling
OVERSAMPLE_DETECTOR = True
OVERSAMPLE_CLASSIFIER = True

# Розбиття та випадковість
VAL_RATIO = 0.2
RANDOM_STATE = 42

# --- Архітектура для перебору ---
# список кількостей прихованих шарів, які тестуємо
HIDDEN_LAYERS_LIST = [1, 2, 3, 4, 5]

# стартовий розмір першого прихованого шару
DET_START_SIZE = 128
CLS_START_SIZE = 128

# мінімальний розмір прихованого шару
MIN_HIDDEN_SIZE = 16

# ініціалізація ваг
DET_INIT = "xavier_normal"       # "xavier_normal", "xavier_uniform", "he"
CLS_INIT = "xavier_normal"

# --- Активації та loss ---
DET_HIDDEN_ACTIVATION = "sigmoid"       # "relu", "gelu", "tanh", "sigmoid"
CLS_HIDDEN_ACTIVATION = "tanh"          # "relu", "gelu", "tanh", "sigmoid"

DET_OUTPUT_ACTIVATION = "sigmoid"
CLS_OUTPUT_ACTIVATION = "softmax"

DET_LOSS = "bce"                 # "bce" або "mse"
CLS_LOSS = "ce"                  # "ce" або "mse"

# --- LayerNorm ---
USE_LAYER_NORM = False
LAYER_NORM_EVERY_K = 0           # 1 = кожен шар, 2 = кожен другий, <=0 = вимкнути LN

# --- Навчання ---
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

# --- Збереження / логування / графіки ---
SAVE_LOGS = True            # зберігати підсумковий CSV-лог по архітектурах
SAVE_PLOTS = True           # будувати та зберігати підсумкові графіки
DEBUG_TRAINING_OUTPUT = True  # показувати епохи у fit(debug_show=...)

# окреме керування побудовою графіків
PLOT_DETECTOR = True        # якщо False — графіки детектора не будуються
PLOT_CLASSIFIER = True      # якщо False — графіки класифікатора не будуються

LOGS_DIR = "logs"
PLOTS_DIR = "plots"

# Папки саме для arch search (використовують LOGS_DIR і PLOTS_DIR)
ARCH_LOGS_DIR = os.path.join(LOGS_DIR, "arch_search")
ARCH_PLOTS_DIR = os.path.join(PLOTS_DIR, "arch_search")

# ============================================================
# Допоміжні функції
# ============================================================


def ensure_dirs():
    """
    Створює (якщо ще не існують) директорії для:
      - загальних логів (LOGS_DIR),
      - загальних графіків (PLOTS_DIR),
      - логів архітектурного пошуку (ARCH_LOGS_DIR),
      - графіків архітектурного пошуку (ARCH_PLOTS_DIR).
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
    Формує список розмірів шарів:
      [input_dim, h1, h2, ... hN, output_dim]

    Кожен наступний прихований шар зменшується приблизно x0.75,
    але не менше, ніж min_hidden.
    Таким чином отримуємо "пірамідальну" архітектуру.
    """
    hidden_sizes = []
    cur = start_size
    for _ in range(n_hidden):
        hidden_sizes.append(cur)
        # наступний шар — 75% від попереднього, але не нижче за min_hidden
        next_size = int(cur * 0.75)
        if next_size < min_hidden:
            next_size = min_hidden
        cur = next_size

    return [input_dim] + hidden_sizes + [output_dim]


def get_config_dict():
    """
    Збирає всі глобальні КАПС-параметри в dict,
    щоб передати у fit(..., config_dict=...) і мати
    повний "зліпок" налаштувань для кожного запуску arch search.
    """
    import sys
    module = sys.modules[__name__]
    cfg = {}
    for name, value in module.__dict__.items():
        # беремо всі змінні у верхньому регістрі (конфіг-настройки)
        if name.isupper() and not name.startswith("_"):
            cfg[name] = value
    return cfg


def measure_inference_speed(net: Network,
                            X: np.ndarray,
                            hidden_activation: str,
                            output_activation: str,
                            n_runs: int = 5):
    """
    Вимірює швидкість inference: скільки зразків/сек
    обробляє мережа на валідації.

    n_runs — скільки разів проганяти повний X через net.predict,
    щоб отримати більш стабільну оцінку часу.
    """
    if X.shape[0] == 0:
        # якщо валідаційний набір пустий — повертаємо 0
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
        # захист від ділення на нуль — якщо час дуже малий/нульовий
        return float("inf"), 0.0
    # samples/sec, загальний час
    return total_samples / elapsed, elapsed


def compute_detector_metrics(y_true: np.ndarray,
                             y_pred_proba: np.ndarray):
    """
    Повертає accuracy, precision, recall, f1 і TN/FP/FN/TP
    для бінарного детектора (аномалія / норма).

    y_true       — істинні мітки (0/1) форми (N,1) або (N,),
    y_pred_proba — ймовірності (вихід sigmoid) форми (N,1) або (N,).
    """
    # приводимо мітки та ймовірності до плоских векторів
    y_true = y_true.reshape(-1).astype(int)
    y_pred_proba = y_pred_proba.reshape(-1)
    # поріг 0.5 → бінарний прогноз
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # компоненти матриці невідповідностей
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))

    # базові бінарні метрики
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
    accuracy та macro-F1 для мультикласового класифікатора.

    y_true       — one-hot матриця (N, C),
    y_pred_proba — ймовірності / логіти (N, C).
    """
    # accuracy_mc і macro_f1_mc всередині самі роблять argmax
    acc = accuracy_mc(y_true, y_pred_proba)
    macro_f1 = macro_f1_mc(y_true, y_pred_proba)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def plot_metric_vs_layers(layers_list, series_dict, title, ylabel, filename):
    """
    Будує графік(и) метрик vs кількість шарів.

    layers_list: список значень n_hidden (кількість прихованих шарів),
    series_dict: {label: [v1, v2, ...]} — для кожної серії метрик,
    title/ylabel: підписи графіка,
    filename: ім'я вихідного PNG в ARCH_PLOTS_DIR.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    for label, values in series_dict.items():
        # малюємо криву для кожної метрики (train/val, тощо)
        plt.plot(layers_list, values, marker="o", label=label)
    plt.xlabel("Кількість прихованих шарів")
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
# Основна логіка архітектурного пошуку
# ============================================================

def main():
    # Створюємо всі потрібні директорії для логів та графіків
    ensure_dirs()
    # Вибираємо бекенд (CPU / GPU)
    set_backend(USE_GPU)

    print("=== СТВОРЕННЯ ДАТАСЕТІВ ДЛЯ ARCH SEARCH ===")
    # Створюємо окремі датасети для детектора і класифікатора
    det_ds, cls_ds = create_dataset(
        csv_path=CSV_PATH,
        scaling_detector=SCALING_DETECTOR,
        scaling_classifier=SCALING_CLASSIFIER,
        oversample_detector=OVERSAMPLE_DETECTOR,
        oversample_classifier=OVERSAMPLE_CLASSIFIER,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
        save_npz=False,  # у архітектурному пошуку .npz не зберігаємо
    )

    # Конфіг з усіма глобальними параметрами для логування у fit()
    config_dict = get_config_dict()
    # Тут будемо накопичувати результати для кожного n_hidden
    summary_rows = []

    # Розміри входу/виходу залишаються сталими, змінюється лише кількість прихованих шарів
    in_det = det_ds.X_train.shape[1]
    in_cls = cls_ds.X_train.shape[1]
    out_det = 1
    out_cls = cls_ds.y_train.shape[1]

    # Перебираємо різну кількість прихованих шарів
    for n_hidden in HIDDEN_LAYERS_LIST:
        print("\n" + "=" * 60)
        print(f"  ТЕСТУЄМО КІЛЬКІСТЬ ПРИХОВАНИХ ШАРІВ: {n_hidden}")
        print("=" * 60)

        # --- формуємо архітектури для детектора та класифікатора ---
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
        # Створюємо мережу-детектор для поточного n_hidden
        det_net = Network(
            det_arch,
            init=DET_INIT,
            use_layernorm=USE_LAYER_NORM,
            ln_every_k=LAYER_NORM_EVERY_K,
        )

        # Навчання детектора + замір часу навчання
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
            save_model=False,              # у arch search ваги не зберігаємо
            model_type=f"detector_L{n_hidden}",
            config_dict=config_dict,
            log_dir=None,                  # окремі логи по епохах тут не пишемо
            plot_metrics=False,            # графіки по епохах вимкнені
            plots_dir=None,
        )
        det_train_time = time.time() - t0
        # останнє значення епохи (останній номер епохи з history)
        det_epochs = int(det_history["epoch"][-1]) if det_history.get("epoch") else 0

        # фінальні train/val loss (по останній епосі)
        det_train_loss_final = np.nan
        det_val_loss_final = np.nan
        if det_history.get("train_loss"):
            det_train_loss_final = float(det_history["train_loss"][-1])
        if det_history.get("val_loss"):
            det_val_loss_final = float(det_history["val_loss"][-1])

        # прогнози для train/val (ймовірності)
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

        # Бінарні метрики для train/val
        det_metrics_val = compute_detector_metrics(det_ds.y_val, y_pred_det_proba_val)
        det_metrics_train = compute_detector_metrics(det_ds.y_train, y_pred_det_proba_train)

        # Швидкість inference на валідації
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
        # Мережа-класифікатор для поточного n_hidden
        cls_net = Network(
            cls_arch,
            init=CLS_INIT,
            use_layernorm=USE_LAYER_NORM,
            ln_every_k=LAYER_NORM_EVERY_K,
        )

        # Навчання класифікатора + замір часу навчання
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

        # Прогнози (one-hot ймовірності) для train/val
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

        # Мультикласові метрики (accuracy + macro-F1) для train/val
        cls_metrics_val = compute_classifier_metrics(cls_ds.y_val, y_pred_cls_proba_val)
        cls_metrics_train = compute_classifier_metrics(cls_ds.y_train, y_pred_cls_proba_train)

        # Швидкість inference на валідації
        cls_speed, _ = measure_inference_speed(
            cls_net,
            cls_ds.X_val,
            hidden_activation=CLS_HIDDEN_ACTIVATION,
            output_activation=CLS_OUTPUT_ACTIVATION,
            n_runs=5,
        )

        # --- фінальні помилки (1 - F1 / 1 - macro-F1) на валідації ---
        det_error_val = 1.0 - float(det_metrics_val["f1"])
        cls_error_val = 1.0 - float(cls_metrics_val["macro_f1"])

        # Записуємо усі важливі показники для поточного n_hidden
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
    # Збереження підсумкового CSV-логу
    # ============================================================
    if SAVE_LOGS:
        # у назві файлу — timestamp, щоб не перезаписувати попередні логи
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(ARCH_LOGS_DIR, f"arch_search_layers_{ts}.csv")

        import csv
        # порядок колонок у підсумковому файлі
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
            # роздільник ";" — зручно відкривати в Excel/LibreOffice
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        print(f"\n[LOG] Підсумковий CSV збережено у {log_path}")
    else:
        print("\n[LOG] SAVE_LOGS = False, CSV-лог не зберігався.")

    # ============================================================
    # Побудова підсумкових графіків
    # ============================================================
    if SAVE_PLOTS:
        # список значень n_hidden, які ми протестували
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
                "DETECTOR: loss (train/val) vs кількість шарів",
                "Loss",
                "detector_loss_vs_layers.png",
            )
            # F1 train/val
            plot_metric_vs_layers(
                layers,
                {"train_F1": det_train_f1_vals, "val_F1": det_val_f1_vals},
                "DETECTOR: F1 (train/val) vs кількість шарів",
                "F1",
                "detector_f1_vs_layers.png",
            )
            # Accuracy train/val
            plot_metric_vs_layers(
                layers,
                {"train_acc": det_train_acc_vals, "val_acc": det_val_acc_vals},
                "DETECTOR: accuracy (train/val) vs кількість шарів",
                "Accuracy",
                "detector_acc_vs_layers.png",
            )
            # Training speed: epochs
            plot_metric_vs_layers(
                layers,
                {"epochs": det_epochs_vals},
                "DETECTOR: кількість епох vs кількість шарів",
                "Епохи",
                "detector_epochs_vs_layers.png",
            )
            # Training speed: time
            plot_metric_vs_layers(
                layers,
                {"train_time_sec": det_time_vals},
                "DETECTOR: час навчання vs кількість шарів",
                "Час навчання, с",
                "detector_time_vs_layers.png",
            )
            # Recognition speed
            plot_metric_vs_layers(
                layers,
                {"inference_speed": det_speed_vals},
                "DETECTOR: швидкість розпізнавання vs кількість шарів",
                "Зразків/сек",
                "detector_speed_vs_layers.png",
            )
            # Error 1 - F1 (val)
            plot_metric_vs_layers(
                layers,
                {"val_error_1-F1": det_error_vals},
                "DETECTOR: похибка (1 - F1) vs кількість шарів",
                "1 - F1 (val)",
                "detector_error_vs_layers.png",
            )

        if PLOT_CLASSIFIER:
            # Loss train/val
            plot_metric_vs_layers(
                layers,
                {"train_loss": cls_train_loss_vals, "val_loss": cls_val_loss_vals},
                "CLASSIFIER: loss (train/val) vs кількість шарів",
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
                "CLASSIFIER: macro-F1 (train/val) vs кількість шарів",
                "macro-F1",
                "classifier_f1_vs_layers.png",
            )
            # Accuracy train/val
            plot_metric_vs_layers(
                layers,
                {"train_acc": cls_train_acc_vals, "val_acc": cls_val_acc_vals},
                "CLASSIFIER: accuracy (train/val) vs кількість шарів",
                "Accuracy",
                "classifier_acc_vs_layers.png",
            )
            # Training speed: epochs
            plot_metric_vs_layers(
                layers,
                {"epochs": cls_epochs_vals},
                "CLASSIFIER: кількість епох vs кількість шарів",
                "Епохи",
                "classifier_epochs_vs_layers.png",
            )
            # Training speed: time
            plot_metric_vs_layers(
                layers,
                {"train_time_sec": cls_time_vals},
                "CLASSIFIER: час навчання vs кількість шарів",
                "Час навчання, с",
                "classifier_time_vs_layers.png",
            )
            # Recognition speed
            plot_metric_vs_layers(
                layers,
                {"inference_speed": cls_speed_vals},
                "CLASSIFIER: швидкість розпізнавання vs кількість шарів",
                "Зразків/сек",
                "classifier_speed_vs_layers.png",
            )
            # Error 1 - macro-F1 (val)
            plot_metric_vs_layers(
                layers,
                {"val_error_1-macroF1": cls_error_vals},
                "CLASSIFIER: похибка (1 - macro-F1) vs кількість шарів",
                "1 - macro-F1 (val)",
                "classifier_error_vs_layers.png",
            )

        print(f"[PLOTS] Графіки збережено у {ARCH_PLOTS_DIR}")
    else:
        print("[PLOTS] SAVE_PLOTS = False, графіки не будувались.")


if __name__ == "__main__":
    # Запускаємо архітектурний пошук тільки при прямому запуску файлу
    main()
