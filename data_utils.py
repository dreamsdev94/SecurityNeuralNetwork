import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =====================================================================
#  Головна функція підготовки датасету
# =====================================================================
def create_dataset(
        csv_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        scaling: str = "zscore",
        return_meta: bool = False
):
    """
    Підготовка датасету:
    1. Завантаження CSV
    2. Видалення рядків з >30% пропусків
    3. Видалення константних стовпців
    4. Видалення дублікатів
    5. Приведення всіх ознак до числового формату
    6. Oversampling меншого класу (по 'type')
    7. Імпутація пропусків медіаною
    8. Нормалізація (Z-score або MinMax)
    9. Розбиття train/val з стратифікацією по 'type'
    10. Кодування цільових міток:
        - бінарна: {-1,+1}
        - мультикласова: one-hot {-1,+1}
    """

    # ================================================================
    # 1. Читання CSV
    # ================================================================
    df = pd.read_csv(csv_path)
    print(f"[load] Loaded shape: {df.shape}")

    # ================================================================
    # 2. Видалення рядків із >30% пропусків
    # ================================================================
    nan_fraction = df.isna().mean(axis=1)
    before = len(df)
    df = df.loc[nan_fraction <= 0.30].copy()
    print(f"[clean] Removed rows >30% NaN: {before - len(df)}")

    # ================================================================
    # 3. Видалення константних стовпців
    # ================================================================
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const_cols)
    print(f"[clean] Removed constant columns: {const_cols}")

    # ================================================================
    # 4. Видалення дублікатів
    # ================================================================
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] Removed duplicates: {before - len(df)}")

    # ================================================================
    # 5. Примусове приведення всіх числових ознак до float
    # ================================================================
    for col in df.columns:
        if col not in ["label", "type"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ================================================================
    #  Перевірка ключових колонок
    # ================================================================
    if "type" not in df.columns:
        raise ValueError("Column 'type' missing in dataset")

    if "label" not in df.columns:
        raise ValueError("Column 'label' missing in dataset")

    # ================================================================
    # 6. OVERSAMPLING меншого класу
    # ================================================================
    counts = df["type"].value_counts()
    max_count = counts.max()

    df_extra = []
    for cls, cnt in counts.items():
        subset = df[df["type"] == cls]
        if cnt < max_count:
            reps = max_count - cnt
            extra = subset.sample(reps, replace=True, random_state=random_state)
            df_extra.append(extra)

    if len(df_extra) > 0:
        df = pd.concat([df] + df_extra, ignore_index=True)

    print("[oversampling] New class distribution:")
    print(df["type"].value_counts())

    # ================================================================
    # 7. Розділення на ознаки і target
    # ================================================================
    feature_cols = [c for c in df.columns if c not in ["label", "type"]]

    # Примусове перетворення на numeric (важливо після oversampling)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    X = df[feature_cols].copy()        # ознаки
    label_bin = df["label"].copy()     # бінарний label
    label_type = df["type"].copy()     # мультикласовий label

    # ================================================================
    # 8. Кодування міток
    # ================================================================
    # 8.1. Бінарний {-1,+1}
    ybin = label_bin.map(lambda x: 1.0 if x == 1 else -1.0).values.reshape(-1, 1)

    # 8.2. One-hot {-1,+1}
    classes = sorted(label_type.unique())
    Y_type = pd.get_dummies(label_type)[classes].values
    Y_pm1 = 2 * Y_type - 1

    # ================================================================
    # 9. Розбиття train/val
    # ================================================================
    Xtr, Xva, ybin_tr, ybin_va, Ytr_type, Yva_type = train_test_split(
        X, ybin, Y_pm1,
        test_size=test_size,
        random_state=random_state,
        stratify=label_type
    )

    print(f"[split] Train: {Xtr.shape}, Val: {Xva.shape}")

    # ================================================================
    # 10. Імпутація та нормалізація
    # ================================================================
    stats = _fit_stats_train(Xtr, scaling=scaling)

    Xtr = _apply_stats(Xtr, stats, scaling=scaling)
    Xva = _apply_stats(Xva, stats, scaling=scaling)

    # ================================================================
    #  Повернення результатів
    # ================================================================
    meta = {
        "classes": classes,
        "kept_columns": feature_cols,
        "scaling": scaling,
        "stats": stats
    }

    if return_meta:
        return (
            Xtr.values.astype(np.float32),
            Xva.values.astype(np.float32),
            Ytr_type.astype(np.float32),
            Yva_type.astype(np.float32),
            meta
        )

    return (
        Xtr.values.astype(np.float32),
        Xva.values.astype(np.float32),
        Ytr_type.astype(np.float32),
        Yva_type.astype(np.float32)
    )


# =====================================================================
#  Обчислення статистики нормалізації
# =====================================================================
def _fit_stats_train(X_df: pd.DataFrame, scaling: str):
    stats = {}

    # імпутація медіаною
    med = X_df.median(numeric_only=True)
    stats["median"] = med

    X_filled = X_df.fillna(med)

    scaling = scaling.lower()

    if scaling == "zscore":
        mu = X_filled.mean(axis=0)
        sd = X_filled.std(axis=0, ddof=0).replace(0, 1.0)
        stats["mean"] = mu
        stats["std"] = sd

    elif scaling == "minmax":
        mins = X_filled.min(axis=0)
        maxs = X_filled.max(axis=0)
        ranges = (maxs - mins).replace(0, 1.0)
        stats["mins"] = mins
        stats["ranges_safe"] = ranges

    else:
        raise ValueError("Unknown scaling method")

    return stats


# =====================================================================
#  Застосування нормалізації
# =====================================================================
def _apply_stats(X_df: pd.DataFrame, stats: dict, scaling: str):
    X = X_df.copy()
    X = X.fillna(stats["median"])

    scaling = scaling.lower()

    if scaling == "zscore":
        return (X - stats["mean"]) / stats["std"]

    elif scaling == "minmax":
        return -1 + (X - stats["mins"]) * 2 / stats["ranges_safe"]

    else:
        raise ValueError("Unknown scaling")
