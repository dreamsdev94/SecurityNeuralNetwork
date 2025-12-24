import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """
    A simple structure for storing a split dataset:
      X_train, y_train — training split
      X_val,   y_val   — validation split
      meta            — a dictionary with additional information (scaling, feature names, etc.)
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    meta: Dict[str, Any]

    def __repr__(self) -> str:
        # Print only array shapes and the list of meta keys
        # to avoid polluting logs with huge structures.
        meta_keys = list(self.meta.keys())
        return (
            "Dataset(\n"
            f"  X_train={self.X_train.shape}, y_train={self.y_train.shape},\n"
            f"  X_val={self.X_val.shape}, y_val={self.y_val.shape},\n"
            f"  meta={meta_keys}\n"
            ")"
        )


def _to_numeric_features(df: pd.DataFrame, exclude_cols=None) -> Tuple[np.ndarray, pd.DataFrame, list]:
    """
    Converts all columns (except exclude_cols) to numeric.
    Non-numeric values → NaN → filled with the median.
    Also removes:
      * columns where all values are NaN
      * constant columns (all values identical)

    Returns:
      X (np.ndarray),
      df_num (DataFrame with features),
      dropped_cols (list of removed columns)
    """
    if exclude_cols is None:
        exclude_cols = []

    # Columns to use as features
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    df_feat = df[feat_cols].copy()

    # Try to cast each feature to float (anything that fails becomes NaN)
    for c in feat_cols:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    dropped_cols = []

    # 1) Drop fully empty columns (all NaN)
    all_nan = df_feat.isna().all(axis=0)
    if all_nan.any():
        for c in df_feat.columns[all_nan]:
            dropped_cols.append(c)
        df_feat = df_feat.loc[:, ~all_nan]

    # 2) Drop constant columns (the same value in all rows)
    const_cols = [c for c in df_feat.columns if df_feat[c].nunique(dropna=True) <= 1]
    if const_cols:
        dropped_cols.extend(const_cols)
        df_feat = df_feat.drop(columns=const_cols)

    # 3) Fill missing values (NaN) with the per-feature median
    df_feat = df_feat.fillna(df_feat.median(numeric_only=True))

    # Final feature matrix
    X = df_feat.values.astype(float)
    return X, df_feat, dropped_cols


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    Splits the data into train/val with a fixed random_state.

    IMPORTANT:
      The same random_state is used for both binary and multiclass targets,
      so the split is aligned (the same indices go to train/val
      for both the detector and the classifier).
    """
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    # Shuffle sample indices
    indices = np.arange(n)
    rng.shuffle(indices)

    # Validation set size
    val_size = int(round(n * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    # Return X_train, y_train, X_val, y_val
    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
    )


def _oversample_binary(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
):
    """
    A simple strategy to balance binary classes by replicating
    the minority class (oversampling).

    y is expected to have shape (N, 1) or (N,).
    """
    # Convert y to a flat vector of labels (0/1)
    y_flat = y.reshape(-1)
    classes, counts = np.unique(y_flat, return_counts=True)
    if len(classes) < 2:
        # If there is only one class — nothing to balance
        return X, y

    max_count = counts.max()
    rng = np.random.default_rng(random_state)

    idx_all = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y_flat == cls)[0]
        if cnt < max_count:
            # If the class is rare — sample extra indices with replacement
            extra = rng.choice(cls_idx, size=max_count - cnt, replace=True)
            new_idx = np.concatenate([cls_idx, extra])
        else:
            # If the class already has max_count samples — keep as is
            new_idx = cls_idx
        idx_all.append(new_idx)

    # Concatenate indices and shuffle again
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


def _oversample_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
):
    """
    Balances multiclass one-hot labels (N, C) by replicating rare classes.

    The algorithm is similar to _oversample_binary, but labels are taken as argmax of
    the one-hot vector.
    """
    # argmax converts the one-hot matrix into a vector of class indices
    labels = np.argmax(y, axis=1)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return X, y

    max_count = counts.max()
    rng = np.random.default_rng(random_state)

    idx_all = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(labels == cls)[0]
        if cnt < max_count:
            # Sample extra elements from the rare class with replacement
            # to reach max_count
            extra = rng.choice(cls_idx, size=max_count - cnt, replace=True)
            new_idx = np.concatenate([cls_idx, extra])
        else:
            new_idx = cls_idx
        idx_all.append(new_idx)

    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


def create_dataset(
    csv_path: str,
    scaling_detector: str = "zscore",
    scaling_classifier: str = "zscore",
    oversample_detector: bool = True,
    oversample_classifier: bool = True,
    val_ratio: float = 0.2,
    random_state: int = 42,
    save_npz: bool = False,
    out_dir: str = "datasets_npz",
):
    """
    Creates two datasets (with the same shared feature matrix X_all):
      1) For the detector (binary) — target column `label`
      2) For the classifier (multiclass) — target column `type`

    Returns: (det_dataset, cls_dataset), where each is a Dataset instance.
    """
    print(f"[LOAD] CSV → {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[SHAPE] initial: {df.shape}")

    # Check required columns
    if "label" not in df.columns or "type" not in df.columns:
        raise ValueError("Expected columns 'label' and 'type' in the CSV file.")

    # === FEATURES (shared for both tasks) ===
    exclude_cols = ["label", "type"]
    X_all, df_feat, dropped_cols = _to_numeric_features(df, exclude_cols=exclude_cols)

    # Store feature names and their medians in meta so they can be used later
    # in realtime scripts without re-parsing the CSV.
    feature_names = df_feat.columns.tolist()
    feature_medians = df_feat.median(numeric_only=True).to_dict()

    if dropped_cols:
        print(f"[CLEAN] remove constant/empty cols: {len(dropped_cols)}")
    else:
        print("[CLEAN] no constant/empty feature columns removed")

    # === Binary target for the detector ===
    # Column label → (N, 1)
    y_bin = df["label"].astype(int).values.reshape(-1, 1)

    # === Multiclass target for the classifier ===
    # Take the 'type' column and fix unique values in order of appearance
    type_series = df["type"].astype(str)
    class_names = list(type_series.unique())  # order of appearance in the dataset
    type_to_idx = {name: i for i, name in enumerate(class_names)}
    y_idx = type_series.map(type_to_idx).values

    # Build the one-hot matrix (N, C)
    n_samples = len(df)
    n_classes = len(class_names)
    y_mc = np.zeros((n_samples, n_classes), dtype=float)
    y_mc[np.arange(n_samples), y_idx] = 1.0

    # === Train/val split (the same one for both tasks) ===
    # First, split X_all and the binary label
    X_tr, y_bin_tr, X_va, y_bin_va = _train_val_split(
        X_all, y_bin, val_ratio=val_ratio, random_state=random_state
    )
    # Then, with the same random_state, split X_all and the multiclass label:
    # indices will be the same, so X_tr/X_va will match for both tasks.
    _, y_mc_tr, _, y_mc_va = _train_val_split(
        X_all, y_mc, val_ratio=val_ratio, random_state=random_state
    )

    # === Scaling function (local) ===
    def _scale(X_train, X_val, mode: str):
        """
        Scales X_train, X_val and also returns a stats dictionary
        with scaling parameters for later reuse.
        """
        if mode == "zscore":
            # Z-score: (x - mean) / std
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True)
            std[std == 0] = 1.0  # guard against division by zero
            X_train_s = (X_train - mean) / std
            X_val_s = (X_val - mean) / std
            return X_train_s, X_val_s, {"mean": mean, "std": std}
        elif mode == "minmax":
            # Min-Max: (x - xmin) / (xmax - xmin)
            xmin = X_train.min(axis=0, keepdims=True)
            xmax = X_train.max(axis=0, keepdims=True)
            denom = xmax - xmin
            denom[denom == 0] = 1.0  # avoid division by 0
            X_train_s = (X_train - xmin) / denom
            X_val_s = (X_val - xmin) / denom
            return X_train_s, X_val_s, {"min": xmin, "max": xmax}
        else:
            # No scaling — return the inputs as is
            return X_train, X_val, {}

    # === DETECTOR (binary task) ===
    # Scale copies of X_tr, X_va (so we do not modify X_all)
    X_det_tr, X_det_va, stats_det = _scale(
        X_tr.copy(), X_va.copy(), scaling_detector
    )
    y_det_tr, y_det_va = y_bin_tr, y_bin_va

    # Optionally balance classes for detector training
    if oversample_detector:
        X_det_tr, y_det_tr = _oversample_binary(
            X_det_tr, y_det_tr, random_state=random_state
        )

    # Detector metadata — everything needed to reproduce preprocessing
    det_meta = {
        "task": "detector",
        "scaling": scaling_detector,
        "val_ratio": val_ratio,
        "csv_path": csv_path,
        "scaling_stats": stats_det,
        "dropped_cols": dropped_cols,
        "feature_names": feature_names,
        "feature_medians": feature_medians,
    }
    det_ds = Dataset(
        X_train=X_det_tr,
        y_train=y_det_tr,
        X_val=X_det_va,
        y_val=y_det_va,
        meta=det_meta,
    )

    # === CLASSIFIER (multiclass task) ===
    X_cls_tr, X_cls_va, stats_cls = _scale(
        X_tr.copy(), X_va.copy(), scaling_classifier
    )
    y_cls_tr, y_cls_va = y_mc_tr, y_mc_va

    # Oversample rare classes for the classifier
    if oversample_classifier:
        X_cls_tr, y_cls_tr = _oversample_multiclass(
            X_cls_tr, y_cls_tr, random_state=random_state
        )

    cls_meta = {
        "task": "classifier",
        "class_names": class_names,
        "scaling": scaling_classifier,
        "val_ratio": val_ratio,
        "csv_path": csv_path,
        "scaling_stats": stats_cls,
        "dropped_cols": dropped_cols,
        "feature_names": feature_names,
        "feature_medians": feature_medians,
    }
    cls_ds = Dataset(
        X_train=X_cls_tr,
        y_train=y_cls_tr,
        X_val=X_cls_va,
        y_val=y_cls_va,
        meta=cls_meta,
    )

    print("\n=== LOADING DATASETS ===")
    print(det_ds)
    print(cls_ds)

    # === (OPTIONAL) SAVE TO .NPZ FILES ===
    if save_npz:
        os.makedirs(out_dir, exist_ok=True)
        det_path = os.path.join(out_dir, "detector_dataset.npz")
        cls_path = os.path.join(out_dir, "classifier_dataset.npz")

        np.savez_compressed(
            det_path,
            X_train=det_ds.X_train,
            y_train=det_ds.y_train,
            X_val=det_ds.X_val,
            y_val=det_ds.y_val,
            meta=det_meta,
        )
        np.savez_compressed(
            cls_path,
            X_train=cls_ds.X_train,
            y_train=cls_ds.y_train,
            X_val=cls_ds.X_val,
            y_val=cls_ds.y_val,
            meta=cls_meta,
        )
        print("\n=== DATASETS SAVED ===")
        print(f"Detector   → {det_path}")
        print(f"Classifier → {cls_path}")

    return det_ds, cls_ds
