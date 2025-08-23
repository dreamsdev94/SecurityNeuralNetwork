# data_utils.py
# -*- coding: utf-8 -*-
"""
Утиліти для підготовки датасету:
- стратифікований поділ за комбінацією (label, type) без reset_index
- очищення та приведення ознак до числового виду
- імпутація пропусків за медіаною ТІЛЬКИ зі статистик train
- відкидання константних колонок за train
- масштабування (MinMax у [-1,1] або Z-score)
- повертає масиви NumPy: X_train, X_val, y_train, y_val

За замовчуванням labels 0/1 мапляться у -1/1
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


# ----------------------------- Стратифікований поділ -----------------------------

"""
Поділ датасету на тренувальний та валідаційний
df - датасет
test_size - частка валідаціного набору
random_state - зерно ГВЧ для відтворювального перемішування
Повертає два дата фрейма
"""
def stratified_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Створення генератора випадкових чисел
    rng = np.random.default_rng(random_state)

    # Формування серії key ка містить значення type
    key = df['type'].astype(str)

    # Сворення списків для зберігання індексів рядків для тренувального та валідаційного наборів
    train_idx = list()
    val_idx = list()

    # Створення словника {type: індекси відповідних рядків}
    # У змінній idx опиняється список індексів рядків для однієї групи type
    for _, idx in key.groupby(key).groups.items():
        idx = list(idx)
        # Перемішування індексів в межах групи
        rng.shuffle(idx)

        # n - кількість рядків у поточній групі type
        n = len(idx)

        # Якщо група має лише один рядок, він відразу потрапляє у train
        if n <= 1:
            train_idx.extend(idx)
            continue

        # Обчислення, скільки елементів групи має потрапити у validation
        n_val = int(round(n * test_size))
        # Гарантування, що у train ш validation буде хоча б один елемент
        n_val = max(1, min(n - 1, n_val))

        # Відправляємо відповідну кількість рядків у train та validation
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    # створення датафреймів train і val, де беруться рядки df за зібраними індексами.
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    return train_df, val_df

"""Очищення текстових колонок і приведення до чисел."""
def _clean_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    # Копіювання датафрейму
    X = X.copy()
    # Вибираємо всі колонки де є текст
    obj = X.select_dtypes(include='object').columns
    if len(obj):
        # ПРибирання пробілів, заміна ком на крапки
        X[obj] = X[obj].apply(lambda s: s.str.strip().str.replace(',', '.', regex=False))
    # Перетворення всіх значень в числа
    X = X.apply(pd.to_numeric, errors='coerce')
    return X

"""Обчислtyyz статистики на train"""
def _fit_stats_train(X_tr_df: pd.DataFrame, scaling: str) -> Dict[str, Any]:

    # Обчислення медіани по кожній колонці
    med = X_tr_df.median(numeric_only=True)
    # Заповнення можливих пропусків медіанами
    X_tr_filled = X_tr_df.fillna(med)

    # відкидаємо константні колонки за train
    # Обчислення мінімумів та максимумів для кожної колонки
    mins = X_tr_filled.min(axis=0)
    maxs = X_tr_filled.max(axis=0)
    # Обчислення для кожної колонки різниці між мінімумом та максимумом
    ranges = maxs - mins
    # СТворення булевого масиву із вказанням неконстантних колонок
    keep_mask = ranges != 0

    # Створення словнику зі статистикою
    stats = {
        "median": med,
        "keep_mask": keep_mask
    }

    # Нормалізація рядку scaling (приведення у нижній регістр)
    scaling = scaling.lower()


    if scaling == "minmax":
        stats.update({
            "mins": mins[keep_mask],
            "ranges": ranges[keep_mask]
        })


    elif scaling == "zscore":
        mu = X_tr_filled.loc[:, keep_mask].mean(axis=0)
        sd = X_tr_filled.loc[:, keep_mask].std(axis=0, ddof=0).replace(0, 1.0)
        stats.update({
            "mean": mu,
            "std": sd
        })
    else:
        raise ValueError("scaling must be 'minmax' or 'zscore'")

    return stats

def _apply_stats(df_part: pd.DataFrame, stats: Dict[str, Any], scaling: str) -> pd.DataFrame:
    """Застосовуємо статистики train до довільної частини (train/val)."""
    X = df_part.copy()
    med = stats["median"]
    keep_mask = stats["keep_mask"]

    X = X.fillna(med)
    X = X.loc[:, keep_mask]

    scaling = scaling.lower()
    if scaling == "minmax":
        mins = stats["mins"]
        ranges = stats["ranges"]
        X = -1 + (X - mins) * 2 / ranges
    elif scaling == "zscore":
        mu = stats["mean"]
        sd = stats["std"]
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
    """
    Зчитує CSV та повертає (X_train, X_val, y_train, y_val) як np.ndarray.
    - Масштабування: 'minmax' у [-1,1] (за train) або 'zscore' (стандартизація за train).
    - labels 0/1 мапляться у -1/1. Якщо вже -1/1 — лишаємо як є.
    - 'type' використовується тільки для стратифікації; з ознак видаляється.
    - Без data leakage: усі статистики рахуються по train і застосовуються до val.

    Якщо return_meta=True — додатково повертає словник метаданих зі статистиками.
    """
    df = pd.read_csv(filename)

    # 1) Стратифікація ДО будь-яких перетворень
    train_df, val_df = stratified_split(df, test_size=test_size, random_state=random_state)

    # 2) Цільові мітки
    def map_target(series: pd.Series) -> np.ndarray:
        uniq = set(series.dropna().unique())
        if uniq <= {0, 1}:
            arr = series.map({0: -1, 1: 1}).to_numpy().reshape(-1, 1)
        else:
            arr = series.to_numpy().reshape(-1, 1)
        return arr.astype(np.float32)

    y_train = map_target(train_df['label'])
    y_val   = map_target(val_df['label'])

    # 3) Ознаки: спільний пул колонок (без label/type)
    feat_cols = df.drop(columns=['label', 'type'], errors='ignore').columns.tolist()

    X_all = _clean_to_numeric(df[feat_cols])
    X_tr_df = X_all.loc[train_df.index].copy()
    X_va_df = X_all.loc[val_df.index].copy()

    # 4) Підганяємо статистики на train і застосовуємо
    stats = _fit_stats_train(X_tr_df, scaling=scaling)
    X_tr = _apply_stats(X_tr_df, stats, scaling=scaling)
    X_va = _apply_stats(X_va_df, stats, scaling=scaling)

    # 5) У NumPy (float32)
    X_train = X_tr.to_numpy(dtype=np.float32)
    X_val   = X_va.to_numpy(dtype=np.float32)

    if not return_meta:
        return X_train, X_val, y_train, y_val

    meta = {
        "scaling": scaling.lower(),
        "columns_all": feat_cols,
        "kept_columns": X_tr.columns.tolist(),
        "stats": {
            # серіалізовані у dict для можливого збереження в JSON
            k: (v.to_dict() if hasattr(v, "to_dict") else v)
            for k, v in stats.items()
        }
    }
    return X_train, X_val, y_train, y_val, meta


# ----------------------------- Аудит (опційно) -----------------------------

def audit_split(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_size: float = 0.2) -> pd.DataFrame:
    """
    Допоміжний звіт розподілу за групами label+type. Повертає DataFrame зі статистикою.
    """
    assert set(train_idx).isdisjoint(set(val_idx)), "Перетин індексів train/val!"
    assert len(train_idx) + len(val_idx) == len(df), "Є втрачені/задвоєні рядки!"

    if 'type' not in df.columns:
        df = df.copy(); df['type'] = 'all'

    full = df.assign(group=df[['label','type']].astype(str).agg('_'.join, axis=1))
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
    # Глобальна частка для довідки
    report.attrs['global_val_ratio_%'] = round(100*len(val_idx)/len(df), 2)
    report.attrs['expected_%'] = int(test_size*100)
    return report