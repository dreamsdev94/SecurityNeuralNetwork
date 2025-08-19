import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def stratified_split(df, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    # якщо 'type' відсутній — зробимо одну групу
    if 'type' not in df.columns:
        df = df.copy()
        df['type'] = 'all'

    # ключ стратифікації: комбінація label+type
    stratify_key = df[['label', 'type']].astype(str).agg('_'.join, axis=1)

    train_idx, val_idx = [], []

    for _, idx in stratify_key.groupby(stratify_key).groups.items():
        idx = list(idx)
        rng.shuffle(idx)

        # скільки віддати у validation (захист від порожніх train/val у малих групах)
        n = len(idx)
        n_val = int(round(n * test_size))
        n_val = max(1, min(n - 1, n_val))  # щонайменше 1 у val і 1 у train, якщо група >1

        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    # ВАЖЛИВО: не скидаємо індекси!
    train_df = df.loc[train_idx]
    val_df   = df.loc[val_idx]

    return train_df, val_df


def create_dataset(filename, test_size=0.2, random_state=42):
    df = pd.read_csv(filename)

    # ціль
    y = df['label']
    if set(y.dropna().unique()) <= {0, 1}:
        y = y.map({0: -1, 1: 1})
    y = y.to_numpy().reshape(-1, 1)

    # ознаки
    X = df.drop(columns=['label', 'type'], errors='ignore').copy()
    obj_cols = X.select_dtypes(include='object').columns
    if len(obj_cols):
        X[obj_cols] = X[obj_cols].apply(lambda s: s.str.strip().str.replace(',', '.', regex=False))
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median(numeric_only=True))

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    keep_mask = ranges != 0
    X = X.loc[:, keep_mask]
    mins = mins[keep_mask]; ranges = ranges[keep_mask]
    X = -1 + (X - mins) * 2 / ranges
    X = X.to_numpy()

    # стратифікований поділ (без reset_index!)
    train_df, val_df = stratified_split(df, test_size=test_size, random_state=random_state)

    # дістаємо рядки за справжніми індексами
    train_idx = train_df.index.to_numpy()
    val_idx   = val_df.index.to_numpy()

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val   = X[val_idx]
    y_val   = y[val_idx]

    return X_train, X_val, y_train, y_val

# def create_dataset(filename):
#     df = pd.read_csv(filename)
#
#     # ціль
#     y = df['label']
#     # якщо мітки 0/1 — зручно для tanh мапнути у -1/1
#     if set(y.dropna().unique()) <= {0, 1}:
#         y = y.map({0: -1, 1: 1})
#     y = y.to_numpy().reshape(-1, 1)
#
#     # ознаки
#     X = df.drop(columns=['label', 'type'], errors='ignore').copy()
#
#     # 1) очистка текстових чисел: прибрати пробіли, замінити коми на крапку
#     obj_cols = X.select_dtypes(include='object').columns
#     if len(obj_cols):
#         X[obj_cols] = X[obj_cols].apply(lambda s: s.str.strip().str.replace(',', '.', regex=False))
#
#     # 2) привести все можливе до чисел; нечислове -> NaN
#     X = X.apply(pd.to_numeric, errors='coerce')
#
#     # 3) якщо з’явились повністю порожні колонки — викидаємо
#     X = X.dropna(axis=1, how='all')
#
#     # 4) заповнити NaN (медіаною по колонці; можна іншу стратегію)
#     X = X.fillna(X.median(numeric_only=True))
#
#     # 5) видалити константні колонки (діапазон 0)
#     mins = X.min(axis=0)
#     maxs = X.max(axis=0)
#     ranges = maxs - mins
#     keep_mask = ranges != 0
#     X = X.loc[:, keep_mask]
#     mins = mins[keep_mask]
#     ranges = ranges[keep_mask]
#
#     # 6) масштабування у [-1, 1]
#     X = -1 + (X - mins) * 2 / ranges
#
#     return X.to_numpy(), y

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
