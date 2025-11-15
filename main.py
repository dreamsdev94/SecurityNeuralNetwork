import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import create_dataset, _apply_stats
from network import Network
import metrics as M


# =====================================================================
#  УТИЛІТА: формування бінарної мітки "загроза/норма"
# =====================================================================
def make_binary_from_types(Y_pm1, classes, normal_name="normal"):
    idx = Y_pm1.argmax(axis=1)
    classes = np.array(classes)
    is_threat = (classes[idx] != normal_name).astype(np.float32)
    return np.where(is_threat > 0.5, 1.0, -1.0).reshape(-1, 1).astype(np.float32)


# =====================================================================
#  АНАЛІЗ ВПЛИВОВИХ ОЗНАК
# =====================================================================
def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    nx = np.sqrt(np.nansum(x * x))
    ny = np.sqrt(np.nansum(y * y))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(np.nansum(x * y) / (nx * ny))


def feature_impact_report(Xtr, Xva, ybin_tr, ybin_va, Ytr_type, Yva_type, feat_names, classes, top_k=15):
    # ---- Бінарний аналіз ----
    corr_tr_bin = []
    corr_va_bin = []
    ytr = ybin_tr.reshape(-1)
    yva = ybin_va.reshape(-1)

    for j in range(Xtr.shape[1]):
        corr_tr_bin.append(abs(_safe_corr(Xtr[:, j], ytr)))
        corr_va_bin.append(abs(_safe_corr(Xva[:, j], yva)))

    df_bin = pd.DataFrame({
        "feature": feat_names,
        "abs_corr_train": corr_tr_bin,
        "abs_corr_val": corr_va_bin
    }).sort_values("abs_corr_val", ascending=False)

    df_bin.to_csv("impact_binary_corr.csv", index=False)

    print("\n[impact] Top binary features:")
    for i, row in df_bin.head(top_k).iterrows():
        print(f"  {row['feature']}: {row['abs_corr_val']:.4f}")

    # ---- Multiclass ----
    C = Ytr_type.shape[1]
    mean_abs = []

    for j in range(Xtr.shape[1]):
        vals = []
        for c in range(C):
            vals.append(abs(_safe_corr(Xva[:, j], Yva_type[:, c])))
        mean_abs.append(np.mean(vals))

    df_mean = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_corr_val": mean_abs
    }).sort_values("mean_abs_corr_val", ascending=False)

    df_mean.to_csv("impact_multiclass_mean_corr.csv", index=False)

    print("\n[impact] Top multiclass features:")
    for i, row in df_mean.head(top_k).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_corr_val']:.4f}")


# =====================================================================
#  ГРАФІКИ
# =====================================================================
def plot_lines(xs, ys_list, labels, title, fname, ylabel=None):
    plt.figure(figsize=(8, 4))
    for y, lbl in zip(ys_list, labels):
        plt.plot(xs, y, label=lbl)
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def save_confusion_image(M, classes, fname, title="Confusion matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(M, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, f"{M[i,j]}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def save_history_csv(hist, fname):
    pd.DataFrame(hist).to_csv(fname, index=False)


# =====================================================================
#  НОВА ЧАСТИНА: ПЕРЕДБАЧЕННЯ НА НОВИХ ДАНИХ
# =====================================================================
def predict_on_new_data(X_new_raw: pd.DataFrame, meta: dict):
    """
    Функція, яка:
    1) Нормалізує нові дані (stats з meta)
    2) Пропускає їх через детектор
    3) Пропускає їх через класифікатор
    4) Повертає два результати:
       - binary: normal/threat
       - multiclass: ім'я класу
    """

    print("\n=== PREDICTION MODE ===")

    # 1) Нормалізація ознак нових даних
    X_norm = _apply_stats(X_new_raw, meta["stats"], meta["scaling"])
    X_norm = X_norm.values.astype(np.float32)

    # 2) Завантаження детектора
    det = Network.load_weights("detector.npz")
    det_out = det.predict(X_norm)
    binary_raw = np.sign(det_out).reshape(-1)

    binary_label = ["normal" if x == -1 else "threat" for x in binary_raw]

    # 3) Завантаження мультикласового класифікатора
    cls = Network.load_weights("classifier.npz")
    cls_out = cls.predict(X_norm)
    cls_idx = cls_out.argmax(axis=1)

    class_names = meta["classes"]
    multiclass_label = [class_names[i] for i in cls_idx]

    return binary_label, multiclass_label


# =====================================================================
#  ГОЛОВНА ЧАСТИНА
# =====================================================================
if __name__ == "__main__":

    CSV_PATH = os.environ.get("CSV_PATH", "Train_Test_Windows_10.csv")

    print("\n=== DATA PREPARATION ===")

    Xtr, Xva, Ytr_type, Yva_type, meta = create_dataset(
        CSV_PATH,
        test_size=0.2,
        random_state=42,
        scaling="zscore",
        return_meta=True
    )

    classes = meta["classes"]
    feat_names = meta["kept_columns"]

    ybin_tr = make_binary_from_types(Ytr_type, classes)
    ybin_va = make_binary_from_types(Yva_type, classes)

    # =================================================================
    # 1) БІНАРНИЙ ДЕТЕКТОР
    # =================================================================
    print("\n=== TRAINING BINARY DETECTOR ===")

    det_layers = [Xtr.shape[1], 64, 1]
    net_det = Network(det_layers, seed=123, final_activation="tanh")

    det_hist = net_det.fit(
        Xtr, ybin_tr,
        Xva, ybin_va,
        max_epochs=500,
        batch_size=128,
        lr=1e-3,
        patience=10,
        monitor="val_acc",
        mode="max",
        optimizer="adam",
        task_hint="binary",
        early_stop_value=0.999,
        early_stop_rounds=2,
        verbose_every=1,
        metrics_module=M
    )

    # ----- Метрики -----
    Yhat_va_det = net_det.predict(Xva)
    det_acc = M.acc_sign(ybin_va, Yhat_va_det)
    p, r, f1 = M.bin_prf(ybin_va, Yhat_va_det)
    tp, fp, fn, tn = M.bin_confusion(ybin_va, Yhat_va_det)

    print(f"[Detector] acc={100*det_acc:.2f}%, P/R/F1={p:.3f}/{r:.3f}/{f1:.3f}")
    print(f"[Detector] Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    plot_lines(det_hist["epoch"],
               [det_hist["train_mse"], det_hist["val_mse"]],
               ["train_mse", "val_mse"],
               "Detector MSE",
               "det_mse.png")

    net_det.save_weights("detector.npz")

    # =================================================================
    # 2) МУЛЬТИКЛАСИФІКАТОР
    # =================================================================
    print("\n=== TRAINING MULTICLASS CLASSIFIER ===")

    C = Ytr_type.shape[1]
    cls_layers = [Xtr.shape[1], 128, 64, C]
    net_cls = Network(cls_layers, seed=123, final_activation="softmax")

    cls_hist = net_cls.fit(
        Xtr, Ytr_type,
        Xva, Yva_type,
        max_epochs=800,
        batch_size=128,
        lr=1e-3,
        patience=12,
        monitor="val_macroF1",
        mode="max",
        optimizer="adam",
        task_hint="multiclass",
        early_stop_value=0.98,
        early_stop_rounds=2,
        verbose_every=1,
        metrics_module=M
    )

    Yhat_va_cls = net_cls.predict(Xva)
    cls_acc = M.acc_argmax(Yva_type, Yhat_va_cls)
    cls_macro = M.macro_f1(Yva_type, Yhat_va_cls)
    cls_top3 = M.top_k_acc(Yva_type, Yhat_va_cls, k=3)

    print(f"[Classifier] acc={100*cls_acc:.2f}%, macroF1={cls_macro:.3f}, top3={100*cls_top3:.2f}%")

    net_cls.save_weights("classifier.npz")

    # =================================================================
    # 3) АНАЛІЗ ОЗНАК
    # =================================================================
    print("\n=== FEATURE IMPACT ANALYSIS ===")
    feature_impact_report(
        Xtr, Xva,
        ybin_tr, ybin_va,
        Ytr_type, Yva_type,
        feat_names,
        classes,
        top_k=15
    )

    # =================================================================
    # 4) ДЕМОНСТРАЦІЯ ПЕРЕДБАЧЕННЯ
    # =================================================================
    print("\n=== DEMO: PREDICT ON A SAMPLE ===")

    # беремо перший валідаційний приклад
    sample_df = pd.DataFrame([Xva[0]], columns=feat_names)

    binary_pred, multiclass_pred = predict_on_new_data(sample_df, meta)

    print("\nPrediction on a sample:")
    print("Binary:", binary_pred[0])
    print("Multiclass:", multiclass_pred[0])

    print("\n=== ALL DONE ===")
