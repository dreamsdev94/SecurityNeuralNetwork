import time
import numpy as np
import matplotlib.pyplot as plt

from data_utils import create_dataset
from metrics import accuracy
from network import Network


CSV_PATH = "Train_Test_Windows_10.csv"


def evaluate_architecture(layers, Xtr, Ytr, Xva, Yva, epochs=40):

    n_out = layers[-1]
    task = "binary" if n_out == 1 else "multiclass"

    # Auto-режими
    if task == "binary":
        hidden = "sigmoid"
        out = "sigmoid"
        loss = "bce"
    else:
        hidden = "tanh"
        out = "softmax"
        loss = "ce"

    n_hidden = len(layers) - 2
    if n_hidden > 7:
        hidden = "relu"

    net = Network(layers)

    t0 = time.time()
    hist = net.fit(
        Xtr, Ytr, Xva, Yva,
        max_epochs=epochs,
        hidden_activation=hidden,
        output_activation=out,
        loss=loss,
        batch_size=128,
        verbose_every=epochs//4
    )
    train_time = time.time() - t0

    # inference speed
    t0 = time.time()
    for _ in range(20):
        _ = net.predict(Xva, hidden_activation=hidden, output_activation=out)
    infer_time = (time.time() - t0) / 20

    yhat = net.predict(Xva, hidden_activation=hidden, output_activation=out)
    score = accuracy(yhat, Yva)

    return score, train_time, infer_time


def test_range(depth_list, Xtr, Ytr, Xva, Yva, base_features):

    metrics = []
    train_speed = []
    infer_speed = []

    for depth in depth_list:
        layer_sizes = np.linspace(base_features, 2, depth+2).astype(int)
        layers = list(layer_sizes)

        print(f"\n===== Testing architecture depth={depth} → {layers} =====")
        score, tr, inf = evaluate_architecture(
            layers, Xtr, Ytr, Xva, Yva
        )

        metrics.append(score)
        train_speed.append(tr)
        infer_speed.append(inf)

    return metrics, train_speed, infer_speed


def plot_results(depths, metrics, train_speed, infer_speed, prefix):

    plt.figure(figsize=(8, 5))
    plt.plot(depths, metrics, marker="o")
    plt.title(f"{prefix} – Accuracy vs Depth")
    plt.xlabel("Hidden layers")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"{prefix}_metrics.png")

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_speed, marker="o")
    plt.title(f"{prefix} – Train time")
    plt.xlabel("Hidden layers")
    plt.ylabel("Seconds")
    plt.grid(True)
    plt.savefig(f"{prefix}_train_speed.png")

    plt.figure(figsize=(8, 5))
    plt.plot(depths, infer_speed, marker="o")
    plt.title(f"{prefix} – Inference time")
    plt.xlabel("Hidden layers")
    plt.ylabel("Seconds")
    plt.grid(True)
    plt.savefig(f"{prefix}_infer_speed.png")


def main():

    Xtr, Xva, Ytr, Yva, meta = create_dataset(
        CSV_PATH, scaling="zscore", return_meta=True
    )

    n_features = Xtr.shape[1]

    # ------------------ Компактні мережі -------------------
    compact_depths = [1, 2, 3, 4, 5]
    c_metrics, c_train, c_infer = test_range(
        compact_depths, Xtr, Ytr, Xva, Yva, n_features
    )
    plot_results(compact_depths, c_metrics, c_train, c_infer, "compact")

    # ------------------ Глибокі мережі -------------------
    deep_depths = [10, 20, 40, 60, 80, 100]
    d_metrics, d_train, d_infer = test_range(
        deep_depths, Xtr, Ytr, Xva, Yva, n_features
    )
    plot_results(deep_depths, d_metrics, d_train, d_infer, "deep")

    print("\nГрафіки збережено у робочій директорії!")


if __name__ == "__main__":
    main()
