import numpy as np
import pandas as pd


class Dataset:
    """
    Універсальна структура датасету:
    містить тренувальні та валідаційні дані.
    """

    def __init__(self, X_train, y_train, X_val, y_val, meta=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.meta = meta or {}

    def __repr__(self):
        return (
            f"Dataset(\n"
            f"  X_train={self.X_train.shape}, "
            f"y_train={self.y_train.shape},\n"
            f"  X_val={self.X_val.shape}, "
            f"y_val={self.y_val.shape},\n"
            f"  meta={list(self.meta.keys())}\n"
            f")"
        )

# ----------------------
# нормалізації
# ----------------------
def normalize_zscore(Xtr, Xva):
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-9
    return (Xtr - mean)/std, (Xva - mean)/std

def normalize_minmax(Xtr, Xva):
    mi = Xtr.min(axis=0)
    ma = Xtr.max(axis=0) + 1e-9
    return (Xtr-mi)/(ma-mi), (Xva-mi)/(ma-mi)


# ----------------------
# легкий SMOTE-style oversampling
# ----------------------
def oversample_balanced(X, y):
    X = np.array(X)
    y = np.array(y)

    unique, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    X_out = []
    y_out = []

    for cls in unique:
        idx = np.where(y == cls)[0]
        cur = X[idx]

        if len(idx) == max_count:
            X_out.append(cur)
            y_out.append(np.full(max_count, cls))
            continue

        reps = max_count - len(idx)
        # SMOTE-style jitter
        noise = np.random.normal(0, 0.01, size=(reps, X.shape[1]))
        synth = cur[np.random.randint(0, len(cur), reps)] + noise

        X_out.append(np.vstack([cur, synth]))
        y_out.append(np.full(max_count, cls))

    return np.vstack(X_out), np.concatenate(y_out)


# ----------------------------------------------------------
# create_dataset (оновлений)
# ----------------------------------------------------------
def create_dataset(
        csv_path,
        test_size=0.2,
        random_state=42,
        scaling_detector="zscore",
        scaling_classifier="zscore",
):
    df = pd.read_csv(csv_path)

    # labels
    df["type"] = df["type"].astype("category")
    y_idx = df["type"].cat.codes.to_numpy()
    class_names = list(df["type"].cat.categories)

    y_bin = df["label"].to_numpy().astype(int)

    # features
    X = df.drop(columns=["type", "label"])
    X = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # drop 30% NaN rows
    good = np.isnan(X).mean(axis=1) <= 0.3
    X = X[good]; y_idx = y_idx[good]; y_bin = y_bin[good]

    # fill NaN
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = col_mean[inds[1]]

    # remove constant columns
    uniq = np.apply_along_axis(lambda c: len(np.unique(c)), 0, X)
    keep_cols = np.where(uniq > 1)[0]
    X = X[:, keep_cols]

    # split
    N = len(X)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(N)

    X = X[idx]
    y_idx = y_idx[idx]
    y_bin = y_bin[idx]

    n_val = int(N*test_size)

    Xtr = X[:-n_val]
    Xva = X[-n_val:]

    ytr_bin = y_bin[:-n_val]
    yva_bin = y_bin[-n_val:]

    ytr_idx = y_idx[:-n_val]
    yva_idx = y_idx[-n_val:]

    # -------------------- detector normalisation --------------------
    if scaling_detector == "zscore":
        Xtr_d, Xva_d = normalize_zscore(Xtr, Xva)
    else:
        Xtr_d, Xva_d = normalize_minmax(Xtr, Xva)

    Xtr_d, ytr_bin_d = oversample_balanced(Xtr_d, ytr_bin)

    # -------------------- classifier normalisation --------------------
    if scaling_classifier == "zscore":
        Xtr_c, Xva_c = normalize_zscore(Xtr, Xva)
    else:
        Xtr_c, Xva_c = normalize_minmax(Xtr, Xva)

    Xtr_c, ytr_idx_bal = oversample_balanced(Xtr_c, ytr_idx)

    # -------------------- one-hot --------------------
    n_classes = len(class_names)
    Ytr = np.eye(n_classes)[ytr_idx_bal]
    Yva = np.eye(n_classes)[yva_idx]

    detector_dataset = Dataset(
        X_train=Xtr_d,
        y_train=ytr_bin_d.reshape(-1,1),
        X_val=Xva_d,
        y_val=yva_bin.reshape(-1,1)
    )

    classifier_dataset = Dataset(
        X_train=Xtr_c,
        y_train=Ytr,
        X_val=Xva_c,
        y_val=Yva
    )

    return detector_dataset, classifier_dataset
