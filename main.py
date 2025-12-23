import os
from datetime import datetime

import numpy as np

from data_utils import create_dataset
from network import Network, set_backend
from metrics import accuracy_mc, macro_f1_mc


# НАЛАШТУВАННЯ В ОДНОМУ МІСЦІ

# --- Загальне ---
USE_GPU = False  # чи пробувати використовувати GPU (CuPy)

CSV_PATH = "Train_Test_Windows_10.csv"

# Нормалізація
SCALING_DETECTOR = "minmax"      # "zscore" або "minmax"
SCALING_CLASSIFIER = "zscore"

# Oversampling
OVERSAMPLE_DETECTOR = True       # балансувати класи для детектора
OVERSAMPLE_CLASSIFIER = True     # балансувати класи для класифікатора

# Архітектура
# кількість прихованих шарів
DET_HIDDEN_LAYERS = 5
CLS_HIDDEN_LAYERS = 5

# стартовий розмір першого прихованого шару
DET_START_SIZE = 128
CLS_START_SIZE = 128

# мінімальний розмір прихованого шару
MIN_HIDDEN_SIZE = 16

# ініціалізація ваг
DET_INIT = "xavier_normal"       # "xavier_normal", "xavier_uniform", "he"
CLS_INIT = "xavier_normal"

# Оптимізатор навчання
OPTIMIZER = None

# --- Активації та loss ---
DET_HIDDEN_ACTIVATION = "relu"       # "relu", "gelu", "tanh", "sigmoid"
CLS_HIDDEN_ACTIVATION = "tanh"       # "relu", "gelu", "tanh", "sigmoid"

DET_OUTPUT_ACTIVATION = "sigmoid"    # бінарний вихід (ймовірність аномалії)
CLS_OUTPUT_ACTIVATION = "softmax"    # мультикласовий вихід (розподіл по класах)

DET_LOSS = "bce"                 # "bce" або "mse"
CLS_LOSS = "ce"                  # "ce" або "mse"

# --- LayerNorm ---
USE_LAYER_NORM = False
LAYER_NORM_EVERY_K = 0           # 1 = кожен шар, 2 = кожен другий, <=0 = вимкнути LN

# Коефіцієнт навчання
DET_LR = 1e-3
CLS_LR = 1e-3

# Розмір батчу
DET_BATCH_SIZE = 256
CLS_BATCH_SIZE = 256

# Максимальна кількість епох
DET_MAX_EPOCHS = 50
CLS_MAX_EPOCHS = 50

# Кількість епох без покращення для ранньої зупинки
DET_PATIENCE = 10
CLS_PATIENCE = 10
MIN_DELTA = 1e-5

# Вимикач ранньої зупинки
EARLY_STOPPING = False

# --- Збереження / логування / графіки ---
SAVE_MODELS = True          # зберігати ваги .npz
SAVE_LOGS = True            # зберігати txt-лог
SAVE_PLOTS = True           # глобально: будувати та зберігати графіки
DEBUG_TRAINING_OUTPUT = True  # виводити епохи в консоль з fit(debug_show=...)

# окреме керування побудовою графіків
PLOT_DETECTOR = True        # якщо False — графіки детектора не будуються
PLOT_CLASSIFIER = True      # якщо False — графіки класифікатора не будуються

MODELS_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"


# ============================================================
# Допоміжні функції
# ============================================================

def get_config_dict():
    """
    Збирає всі глобальні КАПС-налаштування з цього модуля,
    щоб зберегти їх у txt-лог.

    Важливо: ми явно фільтруємо тільки ті константи, які вказані у set(),
    щоб випадково не захопити щось зайве (наприклад, імпортовані з інших модулів).
    """
    import sys
    module = sys.modules[__name__]
    cfg = {}
    for name, value in module.__dict__.items():
        if name.isupper() and not name.startswith("_"):
            # фільтр, щоб не ловити імпортовані CONSTANTS з інших модулів
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
    Побудова списку розмірів шарів типу:
      [input_dim, h1, h2, ..., hN, output_dim]

    Кожен наступний прихований шар зменшується ~x0.75,
    але не менше за min_hidden.

    Це дає "пірамідальну" архітектуру: широкий перший шар, далі звуження.
    """
    sizes = []
    cur = start_size
    for _ in range(n_hidden):
        sizes.append(cur)
        # зменшуємо кількість нейронів, але не опускаємося нижче min_hidden
        next_size = int(cur * 0.75)
        if next_size < min_hidden:
            next_size = min_hidden
        cur = next_size
    # додаємо вхідний та вихідний розміри
    return [input_dim] + sizes + [output_dim]


def ensure_dirs():
    """
    Гарантує наявність директорій для моделей, логів і графіків.
    Якщо прапорці вимкнені — каталоги не створюються.
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
    log_path          — шлях до txt-файлу
    model_type        — "DETECTOR" або "CLASSIFIER"
    layers            — список розмірів шарів
    history           — словник, який повертає Network.fit()
    final_info_lines  — список текстових рядків, які ми виводимо в консоль
    is_binary         — True для детектора, False для класифікатора

    По суті, це "зліпок" усіх налаштувань + історії навчання +
    підсумкових метрик для конкретного запуску.
    """
    cfg = get_config_dict()

    with open(log_path, "w", encoding="utf-8") as f:
        # 1) Повна конфігурація з main.py
        f.write("=== CONFIG (main.py) ===\n")
        for k in sorted(cfg.keys()):
            f.write(f"{k} = {cfg[k]}\n")
        f.write("\n")

        # 2) Інформація про модель
        f.write(f"Model type: {model_type}\n")
        f.write(f"Layers: {layers}\n\n")

        # 3) Історія навчання по епохах
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

        # 4) Підсумкові метрики по найкращій моделі
        f.write("\n=== FINAL METRICS ===\n")
        for line in final_info_lines:
            f.write(line + "\n")


def plot_history(history: dict, base_name: str, out_dir: str):
    """
    Будує та зберігає графіки по history, яке повертає Network.fit().
    Створює кілька .png-файлів у вказаній папці.

    base_name — базова частина імен файлів (без розширення),
    out_dir   — каталог, куди зберігати графіки.
    """
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    if not epochs:
        # якщо history порожня — нічого малювати
        return

    # Графік loss (train / val)
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

    # Бінарний або мультикласовий варіант
    if "val_macro_f1" in history:
        # мультикласовий класифікатор
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
        # бінарний детектор
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
    # 1) GPU / CPU — обрати бекенд для обчислень
    set_backend(USE_GPU)

    # 2) Папки для моделей, логів і графіків
    ensure_dirs()

    # 3) Завантаження та підготовка даних
    # create_dataset:
    #   - читає CSV,
    #   - чистить/перетворює ознаки,
    #   - формує train/val спліт,
    #   - масштабує (zscore/minmax),
    #   - виконує oversampling (якщо потрібно),
    #   - повертає два Dataset: для детектора і для класифікатора.
    det_ds, cls_ds = create_dataset(
        CSV_PATH,
        scaling_detector=SCALING_DETECTOR,
        scaling_classifier=SCALING_CLASSIFIER,
        oversample_detector=OVERSAMPLE_DETECTOR,
        oversample_classifier=OVERSAMPLE_CLASSIFIER,
        save_npz=True,  # паралельно зберігаємо підготовлені датасети у .npz
    )

    # 4) Архітектури
    # розміри входу для обох мереж (кількість ознак)
    in_det = det_ds.X_train.shape[1]
    in_cls = cls_ds.X_train.shape[1]
    # вихід детектора — один нейрон (ймовірність аномалії)
    out_det = 1
    # вихід класифікатора — кількість класів (one-hot)
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

    # Додаємо мітку часу, щоб файли не перезаписували один одного між запусками
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    det_base = f"detector_{DET_HIDDEN_LAYERS}layers_{timestamp}"
    cls_base = f"classifier_{CLS_HIDDEN_LAYERS}layers_{timestamp}"

    print("\n=== TRAINING DETECTOR ===")
    print(f"Detector architecture: {det_arch}")

    # 5) Детектор
    # Створюємо мережу для бінарної задачі
    det_net = Network(
        det_arch,
        init=DET_INIT,
        use_layernorm=USE_LAYER_NORM,
        ln_every_k=LAYER_NORM_EVERY_K,
    )

    # Запускаємо навчання детектора
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

    # Предикт на валідації для фінальних метрик детектора
    y_true_det = det_ds.y_val.reshape(-1).astype(int)
    y_pred_proba_det = det_net.predict(
        det_ds.X_val,
        hidden_activation=DET_HIDDEN_ACTIVATION,
        output_activation=DET_OUTPUT_ACTIVATION,
    ).reshape(-1)
    # Перетворення ймовірностей у бінарні мітки за порогом 0.5
    y_pred_det = (y_pred_proba_det >= 0.5).astype(int)

    # Підрахунок елементів матриці невідповідностей:
    # TN — 0 передбачено як 0, TP — 1 передбачено як 1, FP, FN — помилки.
    tn = int(np.sum((y_pred_det == 0) & (y_true_det == 0)))
    fp = int(np.sum((y_pred_det == 1) & (y_true_det == 0)))
    fn = int(np.sum((y_pred_det == 0) & (y_true_det == 1)))
    tp = int(np.sum((y_pred_det == 1) & (y_true_det == 1)))

    # Обчислення класичних бінарних метрик
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

    # 6) Класифікатор
    print("\n=== TRAINING CLASSIFIER ===")
    print(f"Classifier architecture: {cls_arch}")

    # Мережа для мультикласової задачі
    cls_net = Network(
        cls_arch,
        init=CLS_INIT,
        use_layernorm=USE_LAYER_NORM,
        ln_every_k=LAYER_NORM_EVERY_K,
    )

    # Навчання класифікатора
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

    # Перетворюємо one-hot у індекси класів для істинних і передбачених меток
    y_true_cls = np.argmax(cls_ds.y_val, axis=1)
    y_pred_proba_cls = cls_net.predict(
        cls_ds.X_val,
        hidden_activation=CLS_HIDDEN_ACTIVATION,
        output_activation=CLS_OUTPUT_ACTIVATION,
    )
    y_pred_cls = np.argmax(y_pred_proba_cls, axis=1)

    # Узагальнені мультикласові метрики (accuracy та macro-F1)
    cls_acc = accuracy_mc(y_pred_proba_cls, cls_ds.y_val)
    cls_macro_f1 = macro_f1_mc(y_pred_proba_cls, cls_ds.y_val)

    # Confusion matrix (рядки — істинні класи, стовпці — передбачені)
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

    # 7) Збереження моделей (ваг та архітектури) у .npz
    if SAVE_MODELS:
        det_model_path = os.path.join(MODELS_DIR, det_base + ".npz")
        cls_model_path = os.path.join(MODELS_DIR, cls_base + ".npz")
        det_net.save(det_model_path)
        cls_net.save(cls_model_path)

    # 8) Логи навчання (конфіг + історія + підсумкові метрики)
    if SAVE_LOGS:
        det_log_path = os.path.join(LOGS_DIR, det_base + ".txt")
        cls_log_path = os.path.join(LOGS_DIR, cls_base + ".txt")
        save_log_txt(det_log_path, "DETECTOR", det_arch, det_history, det_lines, True)
        save_log_txt(cls_log_path, "CLASSIFIER", cls_arch, cls_history, cls_lines, False)

    # 9) Графіки по історії навчання
    if SAVE_PLOTS and PLOT_DETECTOR:
        plot_history(det_history, det_base, PLOTS_DIR)
    if SAVE_PLOTS and PLOT_CLASSIFIER:
        plot_history(cls_history, cls_base, PLOTS_DIR)


if __name__ == "__main__":
    # Точка входу — запускаємо main() тільки якщо файл запускається напряму,
    # а не імпортується як модуль.
    main()
