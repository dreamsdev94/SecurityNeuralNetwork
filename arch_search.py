# arch_search.py
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import create_dataset
from network import Network
import metrics as M


# ======================================================================
# Допоміжні функції
# ======================================================================

def run_architecture(layers, Xtr, Ytr, Xva, Yva, max_epochs=300, seed=123):
    """
    Навчає одну архітектуру MLP та повертає словник з метриками.
    Використовується тільки для мультикласової класифікації (softmax).
    """
    print(f"\n[ARCH] Training architecture: {layers}")

    net = Network(layers, seed=seed, final_activation="softmax")

    # рахуємо кількість параметрів
    num_params = 0
    L = len(layers) - 1
    for l in range(1, L + 1):
        W = net.params[f"W{l}"]
        b = net.params[f"b{l}"]
        num_params += W.size + b.size

    t0 = time.time()
    hist = net.fit(
        Xtr, Ytr,
        Xva, Yva,
        max_epochs=max_epochs,
        batch_size=128,
        lr=1e-3,
        patience=10,
        monitor="val_macroF1",
        mode="max",
        optimizer="adam",
        task_hint="multiclass",
        verbose_every=0,
        metrics_module=M
    )
    train_time = time.time() - t0

    # фінальні метрики на валідації
    Yhat_va = net.predict(Xva)
    acc = M.acc_argmax(Yva, Yhat_va)
    macro_f1 = M.macro_f1(Yva, Yhat_va)
    top3 = M.top_k_acc(Yva, Yhat_va, k=3)
    val_loss = hist["val_mse"][-1] if len(hist["val_mse"]) > 0 else np.nan
    epochs = int(hist["epoch"][-1]) if len(hist["epoch"]) > 0 else np.nan

    res = {
        "layers_str": str(layers),
        "input_dim": layers[0],
        "output_dim": layers[-1],
        "hidden_layers": len(layers) - 2,   # без вхідного та вихідного
        "total_layers": len(layers) - 1,    # кількість шарів з вагами
        "num_params": int(num_params),
        "epochs": epochs,
        "train_time_sec": float(train_time),
        "val_loss": float(val_loss),
        "val_acc": float(acc),
        "val_macroF1": float(macro_f1),
        "val_top3": float(top3),
    }
    print(f"[ARCH] done: macroF1={macro_f1:.4f}, acc={acc:.4f}, top3={top3:.4f}")
    return res


def make_deep_layers(input_dim: int, first_hidden: int, class_dim: int):
    """
    Формує "глибоку" архітектуру:
    [input_dim, first_hidden, first_hidden-1, ..., class_dim]

    Приклад: first_hidden = 10, class_dim = 8 ->
      [input_dim, 10, 9, 8]
    """
    hidden = list(range(first_hidden, class_dim, -1))  # 10,9,...,class_dim+1
    hidden.append(class_dim)  # останній прихований = class_dim
    return [input_dim] + hidden   # останній елемент = вихідний шар


def plot_group_metrics(df: pd.DataFrame, title_prefix: str, fname: str):
    """
    Малює один графік для групи архітектур:
      по осі X – кількість прихованих шарів,
      по осі Y – три метрики: acc, macroF1, top3.
    """
    plt.figure(figsize=(8, 4))
    x = df["hidden_layers"].values
    plt.plot(x, df["val_acc"], marker="o", label="val_acc")
    plt.plot(x, df["val_macroF1"], marker="o", label="val_macroF1")
    plt.plot(x, df["val_top3"], marker="o", label="val_top3")
    plt.xlabel("Кількість прихованих шарів")
    plt.ylabel("Значення метрики")
    plt.title(f"{title_prefix}: метрики vs кількість прихованих шарів")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[plot] Saved {fname}")


# ======================================================================
# Головна функція архітектурного пошуку
# ======================================================================

def arch_search(csv_path: str):
    # -------------------------------------------------------------
    # 1. Завантаження та підготовка даних
    # -------------------------------------------------------------
    print("\n=== DATA PREPARATION FOR ARCH SEARCH ===")

    Xtr, Xva, Ytr_pm1, Yva_pm1, meta = create_dataset(
        csv_path,
        test_size=0.2,
        random_state=42,
        scaling="zscore",
        return_meta=True
    )

    classes = meta["classes"]
    print(f"[info] Train/Val shapes: {Xtr.shape}, {Xva.shape}")
    print(f"[info] Classes: {classes}")

    # Переводимо мітки з {-1,+1} у {0,1} для softmax + cross-entropy
    Ytr = (Ytr_pm1 + 1.0) / 2.0
    Yva = (Yva_pm1 + 1.0) / 2.0

    input_dim = Xtr.shape[1]
    class_dim = Ytr.shape[1]

    # -------------------------------------------------------------
    # 2. Компактні архітектури
    # -------------------------------------------------------------
    print("\n=== COMPACT ARCHITECTURES ===")

    compact_archs = [
        [input_dim, 64, class_dim],                        # 1 прихований
        [input_dim, 128, 64, class_dim],                   # 2 прихованих
        [input_dim, 256, 128, 64, class_dim],              # 3 прихованих
        [input_dim, 512, 256, 128, 64, class_dim],         # 4 прихованих
        [input_dim, 512, 256, class_dim],                  # 2 прихованих, але ширших
    ]

    compact_results = []
    for arch in compact_archs:
        res = run_architecture(arch, Xtr, Ytr, Xva, Yva, max_epochs=300, seed=123)
        compact_results.append(res)

    df_compact = pd.DataFrame(compact_results)
    df_compact.to_csv("compact_results.csv", index=False)
    print("[save] compact_results.csv saved")

    plot_group_metrics(df_compact, "Компактні мережі", "compact_metrics_vs_layers.png")

    # -------------------------------------------------------------
    # 3. Глибокі архітектури
    # -------------------------------------------------------------
    print("\n=== DEEP ARCHITECTURES ===")

    # варіанти first_hidden: 10, 50, 100, 200, 500, 1000
    deep_first = [10, 50, 100, 200, 500, 1000]
    deep_results = []

    for n_first in deep_first:
        arch = make_deep_layers(input_dim, n_first, class_dim)
        res = run_architecture(arch, Xtr, Ytr, Xva, Yva, max_epochs=300, seed=123)
        res["first_hidden_size"] = n_first
        deep_results.append(res)

    df_deep = pd.DataFrame(deep_results)
    df_deep.to_csv("deep_results.csv", index=False)
    print("[save] deep_results.csv saved")

    plot_group_metrics(df_deep, "Глибокі мережі", "deep_metrics_vs_layers.png")

    print("\n=== ARCHITECTURE SEARCH FINISHED ===")
    print("Files saved:")
    print("  compact_results.csv, compact_metrics_vs_layers.png")
    print("  deep_results.csv, deep_metrics_vs_layers.png")


# ======================================================================
# Точка входу
# ======================================================================

if __name__ == "__main__":
    csv_path = os.environ.get("CSV_PATH", "Train_Test_Windows_10.csv")
    arch_search(csv_path)
