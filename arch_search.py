import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import create_dataset
from network import Network


# --------------------------------------------------------------
#   НАЛАШТУВАННЯ ГРУП АРХІТЕКТУР
# --------------------------------------------------------------

# Кількість прихованих шарів для "компактних" мереж
COMPACT_HIDDEN_LAYERS = [1, 2, 3, 4, 5]

# Кількість прихованих шарів для "глибоких" мереж
# (останній варіант, який ми ганяли в Colab — до 100 шарів)
# Якщо захочеш, можна розширити до 250, 500, 1000.
DEEP_HIDDEN_LAYERS = [10, 25, 50, 75, 100]

# Початкова ширина (кількість нейронів) для першого прихованого шару
COMPACT_START_WIDTH = 256
DEEP_START_WIDTH = 512


# --------------------------------------------------------------
#   ДОПОМІЖНІ ФУНКЦІЇ
# --------------------------------------------------------------

def build_decreasing_layers(input_dim: int,
                            n_hidden: int,
                            output_dim: int,
                            start_width: int,
                            min_width: int | None = None) -> list[int]:
    """
    Створює список розмірів шарів:
    [input_dim, h1, h2, ..., hL, output_dim],
    де кількість нейронів у прихованих шарах монотонно НЕ зростає
    і плавно спадає від start_width до min_width.
    """
    if min_width is None:
        min_width = max(2, output_dim)

    start_width = max(start_width, min_width)

    if n_hidden <= 0:
        raise ValueError("n_hidden має бути > 0")

    # Якщо шар один — просто start_width
    if n_hidden == 1:
        widths = [start_width]
    else:
        widths = []
        for i in range(n_hidden):
            alpha = i / (n_hidden - 1)  # від 0 до 1
            # Лінійна інтерполяція між start_width та min_width
            w_float = (1.0 - alpha) * start_width + alpha * min_width
            w = int(round(w_float))

            # Гарантуємо, що ширина не зростає
            if widths and w > widths[-1]:
                w = widths[-1]

            widths.append(w)

        # Гарантуємо, що останній прихований шар не менший за min_width
        widths[-1] = max(widths[-1], min_width)

    return [input_dim] + widths + [output_dim]


def accuracy_from_outputs(y_true, y_pred):
    """
    Обчислює accuracy:
    - якщо вихід один (бінарна) → threshold 0.5
    - якщо виходів декілька (мультикласова) → argmax
    """
    y_true = np.asarray(y_true)

    if y_true.ndim == 1 or y_true.shape[1] == 1:
        # Бінарна: y_true в 0/1, y_pred – ймовірності (0..1)
        y_true_bin = y_true.reshape(-1).astype(int)
        y_prob = np.asarray(y_pred).reshape(-1)
        y_pred_bin = (y_prob >= 0.5).astype(int)
        return float(np.mean(y_true_bin == y_pred_bin))

    # Мультикласова: y_true – one-hot (0/1), y_pred – ймовірності (softmax)
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    return float(np.mean(y_true_labels == y_pred_labels))


def run_group(task_name: str,
              X_train: np.ndarray,
              Y_train: np.ndarray,
              X_val: np.ndarray,
              Y_val: np.ndarray,
              group_name: str,
              hidden_layers_list: list[int],
              start_width: int,
              max_epochs: int = 50,
              batch_size: int = 128,
              lr: float = 1e-3) -> list[dict]:
    """
    Проганяє групу архітектур (compact / deep) для однієї задачі
    і повертає список словників з результатами.
    """

    input_dim = X_train.shape[1]
    # Якщо Y має форму (N,) → бінарна; якщо (N, C) → C класів
    if Y_train.ndim == 1:
        output_dim = 1
    else:
        output_dim = Y_train.shape[1]

    results = []

    for n_hidden in hidden_layers_list:
        layers = build_decreasing_layers(
            input_dim=input_dim,
            n_hidden=n_hidden,
            output_dim=output_dim,
            start_width=start_width
        )

        print(
            f"[{task_name}] group={group_name}, hidden_layers={n_hidden}, "
            f"layers={layers}"
        )

        net = Network(layers)

        t0 = time.time()
        hist = net.fit(
            X_train, Y_train,
            X_val,   Y_val,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            loss="auto",          # нехай Network сам вибере BCE/CE + активації
            patience=10,
            verbose_every=max(1, max_epochs // 5),
            reduce_lr_on_plateau=0.5
        )
        dt = time.time() - t0

        # Прогноз на валідації
        Yhat_val = net.predict(X_val)

        acc_val = accuracy_from_outputs(Y_val, Yhat_val)

        result = {
            "task": task_name,
            "group": group_name,
            "n_hidden": n_hidden,
            "layers": layers,
            "val_accuracy": acc_val,
            "final_train_loss": hist["train_loss"][-1],
            "final_val_loss": hist["val_loss"][-1],
            "best_epoch": int(hist["epoch"][np.argmin(hist["val_loss"])]),
            "time_sec": dt,
        }
        results.append(result)

        print(
            f"  → val_acc={acc_val:.4f}, "
            f"best_epoch={result['best_epoch']}, "
            f"time={dt:.1f}s"
        )

    return results


# --------------------------------------------------------------
#   ГОЛОВНА ФУНКЦІЯ ПОШУКУ АРХІТЕКТУР
# --------------------------------------------------------------

def main():
    # Шлях до CSV з логами (як і раніше)
    csv_path = os.environ.get("CSV_PATH", "Train_Test_Windows_10.csv")

    # Створюємо датасет через data_utils.create_dataset
    # Тут використовуємо z-score нормалізацію, як ми налаштували.
    X_train, X_val, y_type_train_pm1, y_type_val_pm1, meta = create_dataset(
        csv_path=csv_path,
        test_size=0.2,
        random_state=42,
        scaling="zscore",
        return_meta=True
    )

    # ----------------- МУЛЬТИКЛАСОВА КЛАСИФІКАЦІЯ -----------------
    # y_type_* з create_dataset йдуть у форматі {-1, +1} (one-vs-rest),
    # переводимо в стандартний one-hot 0/1:
    Ymc_train = (y_type_train_pm1 + 1.0) / 2.0
    Ymc_val   = (y_type_val_pm1   + 1.0) / 2.0

    results_all = []

    print("\n================= MULTICLASS ARCH SEARCH =================")
    res_compact_mc = run_group(
        task_name="multiclass",
        X_train=X_train,
        Y_train=Ymc_train,
        X_val=X_val,
        Y_val=Ymc_val,
        group_name="compact",
        hidden_layers_list=COMPACT_HIDDEN_LAYERS,
        start_width=COMPACT_START_WIDTH,
        max_epochs=50
    )
    res_deep_mc = run_group(
        task_name="multiclass",
        X_train=X_train,
        Y_train=Ymc_train,
        X_val=X_val,
        Y_val=Ymc_val,
        group_name="deep",
        hidden_layers_list=DEEP_HIDDEN_LAYERS,
        start_width=DEEP_START_WIDTH,
        max_epochs=50
    )
    results_all.extend(res_compact_mc)
    results_all.extend(res_deep_mc)

    # ----------------- БІНАРНА КЛАСИФІКАЦІЯ -----------------
    y_label_train_pm1 = meta.get("y_label_train", None)
    y_label_val_pm1   = meta.get("y_label_val", None)

    if y_label_train_pm1 is not None:
        print("\n================= BINARY ARCH SEARCH =================")
        # Переводимо {-1, +1} → {0, 1}
        Ybin_train = (y_label_train_pm1 + 1.0) / 2.0
        Ybin_val   = (y_label_val_pm1   + 1.0) / 2.0

        res_compact_bin = run_group(
            task_name="binary",
            X_train=X_train,
            Y_train=Ybin_train,
            X_val=X_val,
            Y_val=Ybin_val,
            group_name="compact",
            hidden_layers_list=COMPACT_HIDDEN_LAYERS,
            start_width=COMPACT_START_WIDTH,
            max_epochs=50
        )
        res_deep_bin = run_group(
            task_name="binary",
            X_train=X_train,
            Y_train=Ybin_train,
            X_val=X_val,
            Y_val=Ybin_val,
            group_name="deep",
            hidden_layers_list=DEEP_HIDDEN_LAYERS,
            start_width=DEEP_START_WIDTH,
            max_epochs=50
        )
        results_all.extend(res_compact_bin)
        results_all.extend(res_deep_bin)

    # ----------------------------------------------------------
    # ЗБЕРІГАЄМО РЕЗУЛЬТАТИ ТА МАЛЮЄМО ГРАФІКИ
    # ----------------------------------------------------------

    df = pd.DataFrame(results_all)
    df.to_csv("arch_search_results.csv", index=False)
    print("\nРезультати збережено в arch_search_results.csv")

    # Графіки: val_accuracy від кількості прихованих шарів для кожної задачі
    for task in df["task"].unique():
        plt.figure()
        for group in ["compact", "deep"]:
            sub = df[(df["task"] == task) & (df["group"] == group)]
            if sub.empty:
                continue
            sub_sorted = sub.sort_values("n_hidden")
            plt.plot(
                sub_sorted["n_hidden"].to_numpy(),
                sub_sorted["val_accuracy"].to_numpy(),
                marker="o",
                label=f"{group}"
            )

        plt.xlabel("Кількість прихованих шарів")
        plt.ylabel("Accuracy на валідації")
        plt.title(f"Залежність якості від глибини ({task})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"arch_search_{task}.png")

    print("Графіки збережено як arch_search_multiclass.png та arch_search_binary.png (якщо був label).")


if __name__ == "__main__":
    main()
