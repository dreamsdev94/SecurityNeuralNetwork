import pandas as pd
import numpy as np

def create_dataset(filename):
    df = pd.read_csv(filename)

    # ціль
    y = df['label']
    # якщо мітки 0/1 — зручно для tanh мапнути у -1/1
    if set(y.dropna().unique()) <= {0, 1}:
        y = y.map({0: -1, 1: 1})
    y = y.to_numpy().reshape(-1, 1)

    # ознаки
    X = df.drop(columns=['label', 'type'], errors='ignore').copy()

    # 1) очистка текстових чисел: прибрати пробіли, замінити коми на крапку
    obj_cols = X.select_dtypes(include='object').columns
    if len(obj_cols):
        X[obj_cols] = X[obj_cols].apply(lambda s: s.str.strip().str.replace(',', '.', regex=False))

    # 2) привести все можливе до чисел; нечислове -> NaN
    X = X.apply(pd.to_numeric, errors='coerce')

    # 3) якщо з’явились повністю порожні колонки — викидаємо
    X = X.dropna(axis=1, how='all')

    # 4) заповнити NaN (медіаною по колонці; можна іншу стратегію)
    X = X.fillna(X.median(numeric_only=True))

    # 5) видалити константні колонки (діапазон 0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    keep_mask = ranges != 0
    X = X.loc[:, keep_mask]
    mins = mins[keep_mask]
    ranges = ranges[keep_mask]

    # 6) масштабування у [-1, 1]
    X = -1 + (X - mins) * 2 / ranges

    return X.to_numpy(), y

# def create_dataset(filename):
#     df = pd.read_csv(filename)
#
#     sets = df.drop(columns=['label','type'])
#     label_targets = df['label'].to_numpy()
#
#     cols_to_delete = list()
#
#     for col in sets.columns:
#         min_value = sets[col].min()
#         max_value = sets[col].max()
#
#         if min_value == max_value:
#             cols_to_delete.append(col)
#             continue
#
#         sets[col] = -1 + (sets[col] - min_value) * 2 / (max_value - min_value)
#
#     sets = sets.drop(columns=cols_to_delete)
#
#     return sets.to_numpy(), label_targets
