import numpy as np
import pandas as pd


def normalize_zscore(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_train - mean) / std, (X_val - mean) / std


def oversample_minority(X, y):
    unique, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    out_X, out_y = [], []
    for cls, cnt in zip(unique, counts):
        idx = np.where(y == cls)[0]
        reps = int(np.ceil(max_count / cnt))
        new_idx = np.tile(idx, reps)[:max_count]
        out_X.append(X[new_idx])
        out_y.append(np.full(max_count, cls))

    return np.vstack(out_X), np.concatenate(out_y)


def create_dataset(csv_path, test_size=0.2, random_state=42,
                   scaling="zscore", return_meta=False):

    print(f"[LOAD] CSV → {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[SHAPE] initial: {df.shape}")

    col_type = "Type" if "Type" in df.columns else "type"
    df[col_type] = df[col_type].astype("category")

    class_names = list(df[col_type].cat.categories)
    y_idx = df[col_type].cat.codes.to_numpy()

    if "Label" in df.columns or "label" in df.columns:
        col_label = "Label" if "Label" in df.columns else "label"
        y_bin = df[col_label].astype(np.float32).to_numpy()
        y_bin = np.where(y_bin == 1, 1.0, -1.0)
    else:
        y_bin = None

    feature_cols = [c for c in df.columns if c not in ["Type", "type", "Label", "label"]]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    nan_frac = X.isna().mean(axis=1)
    X = X.loc[nan_frac < 0.3].copy()
    X = X.fillna(X.mean())

    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if len(const_cols):
        print(f"[CLEAN] remove constant cols: {len(const_cols)}")
        X = X.drop(columns=const_cols)

    X = X.drop_duplicates()

    y_idx = y_idx[:len(X)]
    if y_bin is not None:
        y_bin = y_bin[:len(X)]

    X = X.to_numpy(np.float32)

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    split = int((1 - test_size) * len(idx))

    tr, va = idx[:split], idx[split:]

    Xtr, Xva = X[tr], X[va]
    ytr_idx, yva_idx = y_idx[tr], y_idx[va]

    if y_bin is not None:
        ytr_bin, yva_bin = y_bin[tr], y_bin[va]
    else:
        ytr_bin = yva_bin = None

    if scaling == "zscore":
        Xtr, Xva = normalize_zscore(Xtr, Xva)
    else:
        raise ValueError("only zscore allowed")

    n_classes = len(class_names)
    ytr_oh = -np.ones((len(ytr_idx), n_classes), np.float32)
    yva_oh = -np.ones((len(yva_idx), n_classes), np.float32)

    ytr_oh[np.arange(len(ytr_idx)), ytr_idx] = 1.0
    yva_oh[np.arange(len(yva_idx)), yva_idx] = 1.0

    if ytr_bin is not None:
        Xtr, ytr_bin = oversample_minority(Xtr, ytr_bin)

    meta = {
        "feature_names": feature_cols,
        "class_names": class_names,
        "y_bin_train": ytr_bin,
        "y_bin_val": yva_bin
    }

    if return_meta:
        return Xtr, Xva, ytr_oh, yva_oh, meta
    else:
        return Xtr, Xva, ytr_oh, yva_oh
