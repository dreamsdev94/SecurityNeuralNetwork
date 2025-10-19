# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import create_dataset
from network import Network
import metrics as M

# ---------- утиліта: бінарний таргет "загроза/норма" з мультикласного Y_pm1 ----------
def make_binary_from_types(Y_pm1, classes, normal_name="normal"):
    """
    Y_pm1: (N, C) у {-1,+1}; клас = argmax по рядку
    +1 -> "threat" (тип != normal), -1 -> "normal"
    """
    idx = Y_pm1.argmax(axis=1)
    classes = np.array(classes)
    is_threat = (classes[idx] != normal_name).astype(np.float32)
    ybin = np.where(is_threat > 0.5, 1.0, -1.0).reshape(-1, 1).astype(np.float32)
    return ybin

# ---------- аналіз впливових ознак (без відкидання; просто звітуємо) ----------
def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    nx = np.sqrt(np.nansum(x*x))
    ny = np.sqrt(np.nansum(y*y))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(np.nansum(x*y) / (nx*ny))

def feature_impact_report(Xtr, Xva, ybin_tr, ybin_va, Ytr_type, Yva_type, feat_names, classes, top_k=15):
    """
    1) Binary: кореляція кожної фічі з міткою (+1/-1)
    2) Multiclass: для кожного класу — кореляція фічі з one-vs-rest-стовпчиком у {-1,+1}
    Звіти зберігаємо в CSV, а ТОП по валідейшну друкуємо.
    """
    # ---- Binary ----
    corr_tr_bin = []
    corr_va_bin = []
    ytr = ybin_tr.reshape(-1); yva = ybin_va.reshape(-1)
    for j in range(Xtr.shape[1]):
        corr_tr_bin.append(abs(_safe_corr(Xtr[:, j], ytr)))
        corr_va_bin.append(abs(_safe_corr(Xva[:, j], yva)))
    df_bin = pd.DataFrame({
        "feature": feat_names,
        "abs_corr_train": corr_tr_bin,
        "abs_corr_val": corr_va_bin
    }).sort_values("abs_corr_val", ascending=False)
    df_bin.to_csv("impact_binary_corr.csv", index=False)

    print("[impact] Top features (binary) by |corr|_val:")
    for i, row in df_bin.head(top_k).iterrows():
        print(f"  {row['feature']}: {row['abs_corr_val']:.4f}")

    # ---- Multiclass (one-vs-rest) ----
    C = Ytr_type.shape[1]
    frames = []
    mean_abs = []
    for j in range(Xtr.shape[1]):
        vals = []
        for c in range(C):
            vals.append(abs(_safe_corr(Xva[:, j], Yva_type[:, c])))
        mean_abs.append(np.mean(vals))
    df_mean = pd.DataFrame({"feature": feat_names, "mean_abs_corr_val": mean_abs}).sort_values("mean_abs_corr_val", ascending=False)
    df_mean.to_csv("impact_multiclass_mean_corr.csv", index=False)

    # ТОП по класах (валідейшн)
    topk_per_class = {}
    for c, cname in enumerate(classes):
        rows = []
        for j in range(Xva.shape[1]):
            rows.append((feat_names[j], abs(_safe_corr(Xva[:, j], Yva_type[:, c]))))
        rows.sort(key=lambda t: t[1], reverse=True)
        topk_per_class[cname] = rows[:top_k]
    # збережемо широким CSV
    max_len = max(len(v) for v in topk_per_class.values())
    out = {}
    for cname, lst in topk_per_class.items():
        names = [f for f, _ in lst] + [""]*(max_len-len(lst))
        corrs = [v for _, v in lst] + [""]*(max_len-len(lst))
        out[f"{cname}_feature"] = names
        out[f"{cname}_|corr|"] = corrs
    pd.DataFrame(out).to_csv("impact_multiclass_top_by_class.csv", index=False)

    print("[impact] Top features by mean |corr| over classes (val):")
    for i, row in df_mean.head(top_k).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_corr_val']:.4f}")

# ---------- графіки ----------
def plot_lines(xs, ys_list, labels, title, fname, ylabel=None):
    plt.figure(figsize=(8,4))
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
    plt.figure(figsize=(6,6))
    plt.imshow(M, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("pred")
    plt.ylabel("true")
    # підписати числа
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, f"{M[i,j]:d}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_history_csv(hist, fname):
    pd.DataFrame(hist).to_csv(fname, index=False)

# ---------- main ----------
if __name__ == "__main__":
    CSV_PATH = os.environ.get("CSV_PATH", "Train_Test_Windows_10.csv")

    # 1) Дані
    Xtr, Xva, Ytr_type, Yva_type, meta = create_dataset(CSV_PATH, test_size=0.2, random_state=42, scaling="minmax", return_meta=True)
    classes = meta["classes"]
    feat_names = meta["kept_columns"]

    print(f"Shapes: {Xtr.shape} {Xva.shape} {Ytr_type.shape} {Yva_type.shape}")
    print(f"Classes: {classes}")

    # Бінарні мітки "threat vs normal"
    ybin_tr = make_binary_from_types(Ytr_type, classes, normal_name="normal")
    ybin_va = make_binary_from_types(Yva_type, classes, normal_name="normal")

    # 2) Детектор (binary)
    det_layers = [Xtr.shape[1], 64, 1]
    net_det = Network(det_layers, seed=123)

    det_hist = net_det.fit(
        Xtr, ybin_tr, Xva, ybin_va,
        max_epochs=500, batch_size=128, lr=1e-3, l2=0.0,
        patience=10, min_delta=1e-6, monitor="val_acc", mode="max",
        optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8, seed=123, task_hint="binary",
        early_stop_value=0.999, early_stop_rounds=2,
        reduce_lr_on_plateau=0.5, lr_patience=5, lr_min=1e-5,
        verbose_every=1,
        metrics_module=M
    )

    # фінальні метрики детектора
    Yhat_va_det = net_det.predict(Xva)
    det_acc = M.acc_sign(ybin_va, Yhat_va_det)
    p, r, f1 = M.bin_prf(ybin_va, Yhat_va_det)
    tp, fp, fn, tn = M.bin_confusion(ybin_va, Yhat_va_det)
    print(f"[Detection] val_acc (sign): {100*det_acc:.2f}% | P/R/F1 {p:.3f}/{r:.3f}/{f1:.3f}")
    print(f"[Detection] Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    # збереження графіків/історії
    plot_lines(det_hist["epoch"], [det_hist["train_mse"], det_hist["val_mse"]],
               ["train_mse", "val_mse"], "Detector MSE", "det_mse.png", ylabel="MSE")
    plot_lines(det_hist["epoch"], [det_hist["val_acc"], det_hist["val_prec"], det_hist["val_rec"], det_hist["val_f1"]],
               ["val_acc", "val_prec", "val_rec", "val_f1"], "Detector val metrics", "det_val_metrics.png", ylabel="value")
    save_history_csv(det_hist, "det_history.csv")
    net_det.save_weights("detector.npz")
    print("Saved: det_mse.png, det_val_metrics.png, det_history.csv")

    # 3) Класифікатор (multiclass)
    C = Ytr_type.shape[1]
    cls_layers = [Xtr.shape[1], 128, 64, C]
    net_cls = Network(cls_layers, seed=123)

    cls_hist = net_cls.fit(
        Xtr, Ytr_type, Xva, Yva_type,
        max_epochs=800, batch_size=128, lr=1e-3, l2=0.0,
        patience=12, min_delta=1e-6, monitor="val_macroF1", mode="max",
        optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8, seed=123, task_hint="multiclass",
        early_stop_value=0.98, early_stop_rounds=2,
        reduce_lr_on_plateau=0.5, lr_patience=6, lr_min=1e-5,
        verbose_every=1,
        metrics_module=M
    )

    Yhat_va_cls = net_cls.predict(Xva)
    cls_acc = M.acc_argmax(Yva_type, Yhat_va_cls)
    cls_macro = M.macro_f1(Yva_type, Yhat_va_cls)
    cls_top3 = M.top_k_acc(Yva_type, Yhat_va_cls, k=3)
    print(f"[Type] val_acc (argmax): {100*cls_acc:.2f}% | macroF1 {cls_macro:.3f} | top3 {100*cls_top3:.2f}%")
    Mmat = M.confusion_matrix_mc(Yva_type, Yhat_va_cls)

    # збереження графіків/історії
    plot_lines(cls_hist["epoch"], [cls_hist["train_mse"], cls_hist["val_mse"]],
               ["train_mse", "val_mse"], "Classifier MSE", "cls_mse.png", ylabel="MSE")
    plot_lines(cls_hist["epoch"], [cls_hist["val_acc"], cls_hist["val_macroF1"], cls_hist["val_top3"]],
               ["val_acc", "val_macroF1", "val_top3"], "Classifier val metrics", "cls_val_metrics.png", ylabel="value")
    save_history_csv(cls_hist, "cls_history.csv")
    save_confusion_image(Mmat, classes, "cls_confusion_matrix.png", title="Confusion (rows=true, cols=pred)")
    net_cls.save_weights("classifier.npz")
    print("Saved: cls_mse.png, cls_val_metrics.png, cls_history.csv")
    print("Saved: cls_confusion_matrix.png")

    # 4) Аналіз впливових ознак (без видалення)
    feature_impact_report(Xtr, Xva, ybin_tr, ybin_va, Ytr_type, Yva_type, feat_names, classes, top_k=15)

    # 5) Приклад донавчання (warm start) — закоментовано для демонстрації
    """
    # Припустимо, у нас з'явився 'new_data.csv' з такою ж структурою
    if os.path.exists("new_data.csv"):
        Xtr2, Xva2, Ytr2, Yva2, _ = create_dataset("new_data.csv", test_size=0.2, random_state=123, scaling=meta["scaling"], return_meta=True)
        # Детектор
        net_det2 = Network.load_weights("detector.npz")
        det_hist2 = net_det2.fit(
            Xtr2, make_binary_from_types(Ytr2, classes), Xva2, make_binary_from_types(Yva2, classes),
            max_epochs=200, batch_size=128, lr=5e-4, patience=8, monitor="val_acc", mode="max",
            seed=777, task_hint="binary", early_stop_value=0.999, early_stop_rounds=1,
            reduce_lr_on_plateau=0.5, lr_patience=4, lr_min=1e-5, warm_start=True, metrics_module=M
        )
        net_det2.save_weights("detector_finetuned.npz")

        # Класифікатор
        net_cls2 = Network.load_weights("classifier.npz")
        cls_hist2 = net_cls2.fit(
            Xtr2, Ytr2, Xva2, Yva2,
            max_epochs=300, batch_size=128, lr=5e-4, patience=10, monitor="val_macroF1", mode="max",
            seed=777, task_hint="multiclass", early_stop_value=0.98, early_stop_rounds=1,
            reduce_lr_on_plateau=0.5, lr_patience=5, lr_min=1e-5, warm_start=True, metrics_module=M
        )
        net_cls2.save_weights("classifier_finetuned.npz")
    """
