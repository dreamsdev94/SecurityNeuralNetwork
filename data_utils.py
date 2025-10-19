# -*- coding: utf-8 -*-
"""
Допоміжні утиліти:
- стратифікований поділ за 'type'
- ознаки -> числові; ІМПУТАЦІЯ медіанами train; МАСШТАБУВАННЯ (MinMax у [-1,1] або Z-score)
- НІЧОГО НЕ ВИДАЛЯЄМО з ознак (навіть константні колонки залишаються)
- ТАРГЕТИ:
    * y_type: one-vs-rest у {-1,+1} з 'type' (мультиклас)
    * y_label: якщо є колонка 'label' — у {-1,+1} (зберігається в meta)
Повертає: X_train, X_val, y_type_train, y_type_val (та meta при return_meta=True)
"""
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def stratified_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифікація за 'type' (категорія)."""
    rng = np.random.default_rng(random_state)
    if 'type' not in df.columns:
        raise ValueError("Немає 'type' — потрібна для мультикласу.")
    key = df['type'].astype(str)
    train_idx, val_idx = [], []
    for _, idx in key.groupby(key).groups.items():
        idx = list(idx); rng.shuffle(idx)
        n = len(idx)
        if n <= 1:
            train_idx.extend(idx); continue
        n_val = int(round(n * test_size))
        n_val = max(1, min(n - 1, n_val))
        val_idx.extend(idx[:n_val]); train_idx.extend(idx[n_val:])
    return df.loc[train_idx], df.loc[val_idx]


def _clean_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    obj = X.select_dtypes(include='object').columns
    if len(obj):
        X[obj] = X[obj].apply(lambda s: s.str.strip().str.replace(',', '.', regex=False))
    X = X.apply(pd.to_numeric, errors='coerce')
    return X


def _fit_stats_train(X_tr_df: pd.DataFrame, scaling: str) -> Dict[str, Any]:
    """Готуємо статистики ТІЛЬКИ з train. НІЧОГО НЕ ВИДАЛЯЄМО."""
    med = X_tr_df.median(numeric_only=True)
    X_tr_filled = X_tr_df.fillna(med)

    mins = X_tr_filled.min(axis=0)
    maxs = X_tr_filled.max(axis=0)
    ranges = maxs - mins
    # без видалення: захистимося від ділення на 0
    ranges_safe = ranges.replace(0, 1.0) if hasattr(ranges, "replace") else np.where(ranges == 0, 1.0, ranges)

    stats = {"median": med, "mins": mins, "maxs": maxs, "ranges_safe": ranges_safe}
    scaling = scaling.lower()
    if scaling == "zscore":
        mu = X_tr_filled.mean(axis=0)
        sd = X_tr_filled.std(axis=0, ddof=0).replace(0, 1.0)
        stats.update({"mean": mu, "std": sd})
    elif scaling != "minmax":
        raise ValueError("scaling must be 'minmax' or 'zscore'")
    return stats


def _apply_stats(df_part: pd.DataFrame, stats: Dict[str, Any], scaling: str) -> pd.DataFrame:
    X = df_part.copy()
    X = X.fillna(stats["median"])
    scaling = scaling.lower()
    if scaling == "minmax":
        mins = stats["mins"]; ranges = stats["ranges_safe"]
        X = -1 + (X - mins) * 2 / ranges
    elif scaling == "zscore":
        mu = stats["mean"]; sd = stats["std"]
        X = (X - mu) / sd
    else:
        raise ValueError("scaling must be 'minmax' or 'zscore'")
    return X


def create_dataset(
    filename: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaling: str = "minmax",
    return_meta: bool = False
):
    df = pd.read_csv(filename)
    if 'type' not in df.columns:
        raise ValueError("У CSV немає 'type' — потрібна як мультикласова ціль.")

    # 1) Поділ
    train_df, val_df = stratified_split(df, test_size=test_size, random_state=random_state)

    # 2) Мультиклас (type) -> one-vs-rest у {-1,+1}
    all_types = pd.Index(df['type'].astype(str).unique()).sort_values()
    type2idx = {t: i for i, t in enumerate(all_types)}
    idx2type = {i: t for t, i in type2idx.items()}
    C = len(type2idx)

    def encode_type_pm1(series: pd.Series) -> np.ndarray:
        idx = series.astype(str).map(type2idx).to_numpy()
        N = idx.shape[0]
        Y = -np.ones((N, C), dtype=np.float32)
        Y[np.arange(N), idx] = 1.0
        return Y

    y_type_train = encode_type_pm1(train_df['type'])
    y_type_val   = encode_type_pm1(val_df['type'])

    # 3) Ознаки (нічого не видаляємо)
    feat_cols = df.drop(columns=['label', 'type'], errors='ignore').columns.tolist()
    X_all = _clean_to_numeric(df[feat_cols])
    X_tr_df = X_all.loc[train_df.index].copy()
    X_va_df = X_all.loc[val_df.index].copy()

    stats = _fit_stats_train(X_tr_df, scaling=scaling)
    X_tr = _apply_stats(X_tr_df, stats, scaling=scaling)
    X_va = _apply_stats(X_va_df, stats, scaling=scaling)

    X_train = X_tr.to_numpy(dtype=np.float32)
    X_val   = X_va.to_numpy(dtype=np.float32)

    if not return_meta:
        return X_train, X_val, y_type_train, y_type_val

    # 4) Додатково: бінарний label (якщо є) у {-1,+1}, повертаємо через meta
    y_label_train = y_label_val = None
    if 'label' in df.columns:
        def encode_label_pm1(series: pd.Series) -> np.ndarray:
            a = pd.to_numeric(series, errors='coerce').fillna(0).to_numpy()
            a = np.where(a > 0, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
            return a
        y_label_train = encode_label_pm1(train_df['label'])
        y_label_val   = encode_label_pm1(val_df['label'])

    meta = dict(
        scaling=scaling.lower(),
        columns_all=feat_cols,
        kept_columns=X_tr.columns.tolist(),  # фактично всі
        classes=list(all_types),
        type2idx=type2idx,
        idx2type=idx2type,
        stats={k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in stats.items()},
        y_label_train=y_label_train,
        y_label_val=y_label_val
    )
    return X_train, X_val, y_type_train, y_type_val, meta
