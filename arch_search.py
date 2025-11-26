# arch_search.py
# Пошук архітектур (глибини) для бінарного детектора та мультикласового класифікатора
# і побудова графіків, як у дипломі.

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import create_dataset
from network import Network
import metrics as M


# ----------------------------------------------------------------------
# Допоміжні функції
# ----------------------------------------------------------------------
def make_binary_from_types(Y_pm1, classes, normal_name="normal"):
    """
    Перетворює мультикласову мітку (one-vs-rest у {-1,+1}) на бінарну:
    +1 -> "загроза" (будь-який тип, крім normal)
    -1 -> "норма"
    """
    idx = Y_pm1.argmax(axis=1)
    classes = np.array(classes)
    is_threat = (classes[idx] != normal_name).astype(np.float32)
    ybin = np.where(is_threat > 0.5, 1.0, -1.0).reshape(-1, 1).astype(np.float32)
    return ybin


def build_compact_layers_detector(in_dim: int, depth: int) -> list:
    """
    Компактні архітектури для детектора:
    in_dim -> (depth прихованих шарів, звуження 64..16) -> 1
    """
    hidden = np.linspace(64, 16, num=depth).astype(int).tolist()
    return [in_dim] + hidden + [1]


def build_compact_layers_classifier(in_dim: int, out_dim: int, depth: int) -> list:
    """
    Компактні архітектури для класифікатора:
    in_dim -> (depth прихованих шарів, звуження 128..32) -> out_dim
    """
    hidden = np.linspace(128, 32, num=depth).astype(int).tolist()
    return [in_dim] + hidden + [out_dim]


def build_deep_layers_detector(in_dim: int, depth: int, width: int = 32) -> list:
    """
    Глибокі архітектури для детектора: багато шарів фіксованої ширини.
    """
    return [in_dim] + [width] * depth + [1]


def build_deep_layers_classifier(in_dim: int, out_dim: int, depth: int, width: int = 64) -> list:
    """
    Глибокі архітектури для класифікатора: багато шарів фіксованої ширини.
    """
    return [in_dim] + [width] * depth + [out_dim]


def measure_inference_speed(net: Network, X_val: np.ndarray) -> tuple[float, float]:
    """
    Повертає (infer_time_sec, cases_per_min).
    """
    t0 = time.time()
    Y_hat = net.predict(X_val)
    infer_time = time.time() - t0
    cases_per_min = (len(X_val) / infer_time) * 60.0 if infer_time > 0 else np.inf
    return infer_time, cases_per_min, Y_hat


# ----------------------------------------------------------------------
# Експерименти для детектора та класифікатора
# ----------------------------------------------------------------------
def run_detector_experiments(Xtr, Xva, ytr, yva, depths, mode="compact"):
    """
    Запускає навчання бінарного детектора для набору глибин.
    Повертає словник з усіма метриками по глибинах.
    """
    results = {
        "depth": [],
        "acc": [],
        "f1": [],
        "epochs": [],
        "train_time_sec": [],
        "infer_time_sec": [],
        "cases_per_min": [],
    }

    in_dim = Xtr.shape[1]

    for d in depths:
        if mode == "compact":
            layers = build_compact_layers_detector(in_dim, d)
        else:
            layers = build_deep_layers_detector(in_dim, d)

        print(f"\n[DETECTOR-{mode}] depth={d}, layers={layers}")

        net = Network(layers, seed=123)

        # --- навчання ---
        t0 = time.time()
        hist = net.fit(
            Xtr, ytr, Xva, yva,
            max_epochs=500,
            batch_size=128,
            lr=1e-3,
            l2=0.0,
            patience=10,
            min_delta=1e-6,
            monitor="val_acc",
            mode="max",
            optimizer="adam",
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            seed=123 + d,
            task_hint="binary",
            early_stop_value=0.999,
            early_stop_rounds=2,
            reduce_lr_on_plateau=0.5,
            lr_patience=5,
            lr_min=1e-5,
            verbose_every=1,
            metrics_module=M,
        )
        train_time = time.time() - t0

        # кількість епох (epoch починається з 0)
        n_epochs = int(hist["epoch"][-1]) + 1 if hist["epoch"] else 0

        # --- інференс + метрики ---
        infer_time, cases_per_min, Y_hat = measure_inference_speed(net, Xva)
        acc = M.acc_sign(yva, Y_hat)
        _, _, f1 = M.bin_prf(yva, Y_hat)

        print(f"[DETECTOR-{mode}] depth={d} | acc={acc:.4f} | F1={f1:.4f} "
              f"| epochs={n_epochs} | train={train_time:.2f}s | infer={infer_time:.4f}s")

        results["depth"].append(d)
        results["acc"].append(acc)
        results["f1"].append(f1)
        results["epochs"].append(n_epochs)
        results["train_time_sec"].append(train_time)
        results["infer_time_sec"].append(infer_time)
        results["cases_per_min"].append(cases_per_min)

    return results


def run_classifier_experiments(Xtr, Xva, Ytr, Yva, depths, mode="compact"):
    """
    Запускає навчання мультикласового класифікатора для набору глибин.
    Повертає словник з метриками по глибинах.
    """
    results = {
        "depth": [],
        "acc": [],
        "macroF1": [],
        "top3": [],
        "epochs": [],
        "train_time_sec": [],
        "infer_time_sec": [],
        "cases_per_min": [],
    }

    in_dim = Xtr.shape[1]
    C = Ytr.shape[1]

    for d in depths:
        if mode == "compact":
            layers = build_compact_layers_classifier(in_dim, C, d)
        else:
            layers = build_deep_layers_classifier(in_dim, C, d)

        print(f"\n[CLASSIFIER-{mode}] depth={d}, layers={layers}")

        net = Network(layers, seed=321)

        # --- навчання ---
        t0 = time.time()
        hist = net.fit(
            Xtr, Ytr, Xva, Yva,
            max_epochs=800,
            batch_size=128,
            lr=1e-3,
            l2=0.0,
            patience=12,
            min_delta=1e-6,
            monitor="val_macroF1",
            mode="max",
            optimizer="adam",
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            seed=321 + d,
            task_hint="multiclass",
            early_stop_value=0.98,
            early_stop_rounds=2,
            reduce_lr_on_plateau=0.5,
            lr_patience=6,
            lr_min=1e-5,
            verbose_every=1,
            metrics_module=M,
        )
        train_time = time.time() - t0
        n_epochs = int(hist["epoch"][-1]) + 1 if hist["epoch"] else 0

        # --- інференс + метрики ---
        infer_time, cases_per_min, Y_hat = measure_inference_speed(net, Xva)
        acc = M.acc_argmax(Yva, Y_hat)
        macroF1 = M.macro_f1(Yva, Y_hat)
        top3 = M.top_k_acc(Yva, Y_hat, k=3)

        print(f"[CLASSIFIER-{mode}] depth={d} | acc={acc:.4f} | macroF1={macroF1:.4f} "
              f"| top3={top3:.4f} | epochs={n_epochs} | train={train_time:.2f}s | infer={infer_time:.4f}s")

        results["depth"].append(d)
        results["acc"].append(acc)
        results["macroF1"].append(macroF1)
        results["top3"].append(top3)
        results["epochs"].append(n_epochs)
        results["train_time_sec"].append(train_time)
        results["infer_time_sec"].append(infer_time)
        results["cases_per_min"].append(cases_per_min)

    return results


# ----------------------------------------------------------------------
# Побудова графіків (як на скрінах)
# ----------------------------------------------------------------------
def plot_infer_speed(depths, cases_per_min, title, label, fname):
    plt.figure(figsize=(8, 4))
    plt.plot(depths, cases_per_min, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Кількість прихованих шарів")
    plt.ylabel("Кількість оброблених кейсів за хвилину")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_metrics_compact_classifier(res, fname):
    plt.figure(figsize=(8, 4))
    d = res["depth"]
    plt.plot(d, res["acc"], marker="o", label="acc (classifier)")
    plt.plot(d, res["macroF1"], marker="s", label="macroF1 (classifier)")
    plt.plot(d, res["top3"], marker="^", label="top3 (classifier)")
    plt.title("Метріки мультикласового класифікатора vs глибина (компактні мережі)")
    plt.xlabel("Кількість прихованих шарів")
    plt.ylabel("Значення метрик")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_metrics_compact_detector(res, fname):
    plt.figure(figsize=(8, 4))
    d = res["depth"]
    plt.plot(d, res["acc"], marker="o", label="acc (detector)")
    plt.plot(d, res["f1"], marker="s", label="F1 (detector)")
    plt.title("Метріки бінарного детектора vs глибина (компактні мережі)")
    plt.xlabel("Кількість прихованих шарів")
    plt.ylabel("Значення метрик")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_train_speed(res, title, fname, what="classifier"):
    d = res["depth"]
    epochs = res["epochs"]
    tsec = res["train_time_sec"]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(d, epochs, marker="o", label=f"epochs ({what})")
    ax1.set_xlabel("Кількість прихованих шарів")
    ax1.set_ylabel("Кількість епох")

    ax2 = ax1.twinx()
    ax2.plot(d, tsec, marker="s", linestyle="--", label=f"train_time_sec ({what})")
    ax2.set_ylabel("Час навчання, с")

    fig.suptitle(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_metrics_deep_classifier(res, fname):
    plt.figure(figsize=(8, 4))
    d = res["depth"]
    plt.plot(d, res["acc"], marker="o", label="acc (classifier)")
    plt.plot(d, res["macroF1"], marker="s", label="macroF1 (classifier)")
    plt.plot(d, res["top3"], marker="^", label="top3 (classifier)")
    plt.title("Метріки мультикласового класифікатора vs глибина (глибокі мережі)")
    plt.xlabel("Кількість прихованих шарів")
    plt.ylabel("Значення метрик")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    CSV_PATH = os.environ.get("CSV_PATH", "Train_Test_Windows_10.csv")

    # 1) Дані
    Xtr, Xva, Ytr_type, Yva_type, meta = create_dataset(
        CSV_PATH,
        test_size=0.2,
        random_state=42,
        scaling="minmax",
        return_meta=True,
    )
    classes = meta["classes"]

    print(f"[DATA] Xtr={Xtr.shape}, Xva={Xva.shape}, Ytr_type={Ytr_type.shape}, Yva_type={Yva_type.shape}")
    print(f"[DATA] classes: {classes}")

    # Бінарні мітки для детектора
    ybin_tr = make_binary_from_types(Ytr_type, classes, normal_name="normal")
    ybin_va = make_binary_from_types(Yva_type, classes, normal_name="normal")

    # ------------------------------------------------------------------
    # 2) Компактні мережі (depth = 1..5)
    # ------------------------------------------------------------------
    depths_compact = [1, 2, 3, 4, 5]

    det_comp = run_detector_experiments(Xtr, Xva, ybin_tr, ybin_va, depths_compact, mode="compact")
    cls_comp = run_classifier_experiments(Xtr, Xva, Ytr_type, Yva_type, depths_compact, mode="compact")

    # Зберігаємо таблиці
    pd.DataFrame(det_comp).to_csv("arch_compact_detector.csv", index=False)
    pd.DataFrame(cls_comp).to_csv("arch_compact_classifier.csv", index=False)

    # Графіки для компактних
    plot_infer_speed(
        det_comp["depth"], det_comp["cases_per_min"],
        "Швидкість розпізнавання бінарного детектора (компактні мережі)",
        "cases/min (detector)",
        "detector_infer_speed_compact.png",
    )
    plot_metrics_compact_detector(det_comp, "detector_metrics_vs_depth_compact.png")
    plot_train_speed(
        det_comp,
        "Швидкість навчання бінарного детектора (компактні мережі)",
        "detector_train_speed_compact.png",
        what="detector",
    )

    plot_infer_speed(
        cls_comp["depth"], cls_comp["cases_per_min"],
        "Швидкість розпізнавання мультикласового класифікатора (компактні мережі)",
        "cases/min (classifier)",
        "classifier_infer_speed_compact.png",
    )
    plot_metrics_compact_classifier(cls_comp, "classifier_metrics_vs_depth_compact.png")
    plot_train_speed(
        cls_comp,
        "Швидкість навчання мультикласового класифікатора (компактні мережі)",
        "classifier_train_speed_compact.png",
        what="classifier",
    )

    # ------------------------------------------------------------------
    # 3) Глибокі мережі (багато шарів)
    # ------------------------------------------------------------------
    depths_deep = [10, 25, 50, 75, 100]

    det_deep = run_detector_experiments(Xtr, Xva, ybin_tr, ybin_va, depths_deep, mode="deep")
    cls_deep = run_classifier_experiments(Xtr, Xva, Ytr_type, Yva_type, depths_deep, mode="deep")

    pd.DataFrame(det_deep).to_csv("arch_deep_detector.csv", index=False)
    pd.DataFrame(cls_deep).to_csv("arch_deep_classifier.csv", index=False)

    # Для детектора (глибокі) на твоїх графіках – тільки швидкість інференсу
    plot_infer_speed(
        det_deep["depth"], det_deep["cases_per_min"],
        "Швидкість розпізнавання бінарного детектора (глибокі мережі)",
        "cases/min (detector)",
        "detector_infer_speed_deep.png",
    )

    # Для класифікатора (глибокі) – і швидкість, і метрики, і швидкість навчання
    plot_infer_speed(
        cls_deep["depth"], cls_deep["cases_per_min"],
        "Швидкість розпізнавання мультикласового класифікатора (глибокі мережі)",
        "cases/min (classifier)",
        "classifier_infer_speed_deep.png",
    )
    plot_metrics_deep_classifier(cls_deep, "classifier_metrics_vs_depth_deep.png")
    plot_train_speed(
        cls_deep,
        "Швидкість навчання мультикласового класифікатора (глибокі мережі)",
        "classifier_train_speed_deep.png",
        what="classifier",
    )

    print("\n[DONE] Збережено CSV та всі графіки для compact/deep детектора й класифікатора.")


if __name__ == "__main__":
    main()
