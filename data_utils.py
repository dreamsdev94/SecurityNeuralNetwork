# data_utils.py
# -*- coding: utf-8 -*-
"""
Утиліти для підготовки датасету під мультиклас (type):
- стратифікований поділ за колонкою 'type' (без reset_index)
- ознаки -> числові, імпутація медіанами train, відкидання константних колонок
- масштабування (MinMax у [-1,1] або Z-score) за статистиками train
- ТАРГЕТ: one-vs-rest у {-1, 1} з колонки 'type' (мультиклас)
- Повертає: X_train, X_val, y_train, y_val (NumPy)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


# ----------------------------- Стратифікований поділ -----------------------------

def stratified_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифікація за 'type' (категорія аномалії)."""
    rng = np.random.default_rng(random_state)
    if 'type' not in df.columns:
        raise ValueError("У датасеті відсутня колонка 'type' — потрібна для мультикласової цілі.")

    key = df['type'].astype(str)
    train_idx, val_idx = [], []

    for _, idx in key.groupby(key).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        if n <= 1:
            train_idx.extend(idx)
            continue
        n_val = int(round(n * test_size))
        n_val = max(1, min(n - 1, n_val))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    return train_df, val_df


# ----------------------------- Очистка та статистики -----------------------------

def _clean_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    obj = X.select_dtypes(include='object').columns
    if len(obj):
        X[obj] = X[obj].apply(lambda s: s.str.strip().str.replace(',', '.', regex=False))
    X = X.apply(pd.to_numeric, errors='coerce')
    return X


def _fit_stats_train(X_tr_df: pd.DataFrame, scaling: str) -> Dict[str, Any]:
    med = X_tr_df.median(numeric_only=True)
    X_tr_filled = X_tr_df.fillna(med)

    mins = X_tr_filled.min(axis=0)
    maxs = X_tr_filled.max(axis=0)
    ranges = maxs - mins
    keep_mask = ranges != 0

    stats = {"median": med, "keep_mask": keep_mask}
    scaling = scaling.lower()

    if scaling == "minmax":
        stats.update({"mins": mins[keep_mask], "ranges": ranges[keep_mask]})
    elif scaling == "zscore":
        mu = X_tr_filled.loc[:, keep_mask].mean(axis=0)
        sd = X_tr_filled.loc[:, keep_mask].std(axis=0, ddof=0).replace(0, 1.0)
        stats.update({"mean": mu, "std": sd})
    else:
        raise ValueError("scaling must be 'minmax' or 'zscore'")

    return stats


def _apply_stats(df_part: pd.DataFrame, stats: Dict[str, Any], scaling: str) -> pd.DataFrame:
    X = df_part.copy()
    med = stats["median"]
    keep_mask = stats["keep_mask"]

    X = X.fillna(med)
    X = X.loc[:, keep_mask]

    scaling = scaling.lower()
    if scaling == "minmax":
        mins = stats["mins"]; ranges = stats["ranges"]
        X = -1 + (X - mins) * 2 / ranges
    elif scaling == "zscore":
        mu = stats["mean"]; sd = stats["std"]
        X = (X - mu) / sd
    else:
        raise ValueError("scaling must be 'minmax' or 'zscore'")

    return X


# ----------------------------- Основна функція -----------------------------

def create_dataset(
    filename: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaling: str = "minmax",
    return_meta: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:

    df = pd.read_csv(filename)

    if 'type' not in df.columns:
        raise ValueError("У CSV немає колонки 'type' — вона потрібна як ціль (мультиклас).")

    # Поділ на train/val
    train_df, val_df = stratified_split(df, test_size=test_size, random_state=random_state)

    # ---- Мультикласовий таргет з 'type' ----
    # визначимо всі унікальні типи на всьому датасеті (стабільний словник класів)
    all_types = pd.Index(df['type'].astype(str).unique())
    all_types = all_types.sort_values()  # стабільний порядок
    type2idx = {t: i for i, t in enumerate(all_types)}
    idx2type = {i: t for t, i in type2idx.items()}
    C = len(type2idx)

    def encode_type_to_pm1(series: pd.Series) -> np.ndarray:
        """One-vs-rest у {-1, 1} для списку міток 'type'."""
        idx = series.astype(str).map(type2idx).to_numpy()
        N = idx.shape[0]
        Y = -np.ones((N, C), dtype=np.float32)
        Y[np.arange(N), idx] = 1.0
        return Y

    y_train = encode_type_to_pm1(train_df['type'])
    y_val   = encode_type_to_pm1(val_df['type'])

    # ---- Ознаки ----
    # Викидаємо з фіч і 'label', і 'type' на випадок наявності обох.
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
        return X_train, X_val, y_train, y_val

    meta = {
        "scaling": scaling.lower(),
        "columns_all": feat_cols,
        "kept_columns": X_tr.columns.tolist(),
        "classes": list(all_types),
        "type2idx": type2idx,
        "idx2type": idx2type,
        "stats": {
            k: (v.to_dict() if hasattr(v, "to_dict") else v)
            for k, v in stats.items()
        }
    }
    return X_train, X_val, y_train, y_val, meta


# ----------------------------- Аудит (опційно) -----------------------------

def audit_split(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_size: float = 0.2) -> pd.DataFrame:
    """Допоміжний звіт розподілу за групами label+type."""
    assert set(train_idx).isdisjoint(set(val_idx)), "Перетин індексів train/val!"
    assert len(train_idx) + len(val_idx) == len(df), "Є втрачені/задвоєні рядки!"

    if 'type' not in df.columns:
        df = df.copy(); df['type'] = 'all'

    full = df.assign(group=df[['label','type']].astype(str).agg('_'.join, axis=1) if 'label' in df.columns
                     else df['type'].astype(str))
    tr = full.loc[train_idx]; va = full.loc[val_idx]

    g_full = full['group'].value_counts().sort_index()
    g_tr   = tr['group'].value_counts().reindex(g_full.index, fill_value=0)
    g_va   = va['group'].value_counts().reindex(g_full.index, fill_value=0)

    report = pd.DataFrame({
        'full': g_full,
        'train': g_tr,
        'val': g_va,
        'val_ratio(%)': (100.0 * g_va / g_full).round(1)
    })
    report.attrs['global_val_ratio_%'] = round(100*len(val_idx)/len(df), 2)
    report.attrs['expected_%'] = int(test_size*100)
    return report