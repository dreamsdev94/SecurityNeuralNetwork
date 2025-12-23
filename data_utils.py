import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """
    Проста структура для зберігання розбитого датасету:
      X_train, y_train — навчальна вибірка
      X_val,   y_val   — валідаційна вибірка
      meta            — словник з додатковою інформацією (масштабування, назви ознак тощо)
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    meta: Dict[str, Any]

    def __repr__(self) -> str:
        # Виводимо тільки розміри масивів та список ключів meta,
        # щоб не засмічувати лог величезними структурами.
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
    Перетворення всіх колонок (крім exclude_cols) у числові.
    Нечислові значення → NaN → заповнюємо медіаною.
    Також видаляємо:
      * колонки, де всі значення NaN
      * константні колонки (усі значення однакові)
    Повертаємо:
      X (np.ndarray),
      df_num (DataFrame з ознаками),
      dropped_cols (список видалених колонок)
    """
    if exclude_cols is None:
        exclude_cols = []

    # Колонки, які будемо використовувати як ознаки
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    df_feat = df[feat_cols].copy()

    # Пробуємо привести кожну ознаку до float (усе, що не вийшло, стане NaN)
    for c in feat_cols:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    dropped_cols = []

    # 1) Викидаємо повністю порожні (усі NaN) колонки
    all_nan = df_feat.isna().all(axis=0)
    if all_nan.any():
        for c in df_feat.columns[all_nan]:
            dropped_cols.append(c)
        df_feat = df_feat.loc[:, ~all_nan]

    # 2) Викидаємо константні колонки (одне й те саме значення у всіх рядках)
    const_cols = [c for c in df_feat.columns if df_feat[c].nunique(dropna=True) <= 1]
    if const_cols:
        dropped_cols.extend(const_cols)
        df_feat = df_feat.drop(columns=const_cols)

    # 3) Заповнюємо пропуски (NaN) медіаною по кожній ознаці
    df_feat = df_feat.fillna(df_feat.median(numeric_only=True))

    # Фінальна матриця ознак
    X = df_feat.values.astype(float)
    return X, df_feat, dropped_cols


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    Розбиває дані на train/val з фіксованим random_state.

    ВАЖЛИВО:
      Використовується той самий random_state і для бінарної, і для мультикласової
      цілі, тому розбиття узгоджене (одні й ті самі індекси потрапляють у train/val
      і для детектора, і для класифікатора).
    """
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    # Перемішуємо індекси об'єктів
    indices = np.arange(n)
    rng.shuffle(indices)

    # Розмір валідаційної частини
    val_size = int(round(n * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    # Повертаємо X_train, y_train, X_val, y_val
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
    Проста стратегія балансування бінарних класів шляхом реплікації
    меншого класу (oversampling).

    y очікується форми (N, 1) або (N,).
    """
    # Перетворюємо y у плоский вектор міток (0/1)
    y_flat = y.reshape(-1)
    classes, counts = np.unique(y_flat, return_counts=True)
    if len(classes) < 2:
        # Якщо всього один клас — нічого балансувати
        return X, y

    max_count = counts.max()
    rng = np.random.default_rng(random_state)

    idx_all = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y_flat == cls)[0]
        if cnt < max_count:
            # Якщо клас рідкісний — добираємо випадкові індекси з нього із повторенням
            extra = rng.choice(cls_idx, size=max_count - cnt, replace=True)
            new_idx = np.concatenate([cls_idx, extra])
        else:
            # Якщо клас уже має max_count зразків — лишаємо як є
            new_idx = cls_idx
        idx_all.append(new_idx)

    # Об'єднуємо індекси і знову перемішуємо
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


def _oversample_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
):
    """
    Балансування мультикласу one-hot (N, C) шляхом реплікації рідкісних класів.

    Алгоритм аналогічний _oversample_binary, але мітки беруться як argmax по
    one-hot вектору.
    """
    # argmax перетворює one-hot матрицю на вектор індексів класів
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
            # Добираємо елементи рідкісного класу з повторенням,
            # щоб довести до max_count
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
    Створює два датасети (з однаковими ознаками X_all):
      1) Для детектора (бінарний) — цільова колонка `label`
      2) Для класифікатора (мультиклас) — цільова колонка `type`

    Повертає: (det_dataset, cls_dataset), де кожен — екземпляр Dataset.
    """
    print(f"[LOAD] CSV → {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[SHAPE] initial: {df.shape}")

    # Перевіряємо наявність обов'язкових колонок
    if "label" not in df.columns or "type" not in df.columns:
        raise ValueError("Очікуються колонки 'label' і 'type' у CSV-файлі.")

    # === ОЗНАКИ (спільні для обох задач) ===
    exclude_cols = ["label", "type"]
    X_all, df_feat, dropped_cols = _to_numeric_features(df, exclude_cols=exclude_cols)

    # Імена ознак та їх медіани зберігаємо в meta, щоб потім
    # їх можна було використовувати в realtime-скриптах без повторного парсингу CSV.
    feature_names = df_feat.columns.tolist()
    feature_medians = df_feat.median(numeric_only=True).to_dict()

    if dropped_cols:
        print(f"[CLEAN] remove constant/empty cols: {len(dropped_cols)}")
    else:
        print("[CLEAN] no constant/empty feature columns removed")

    # === Бінарна ціль для детектора ===
    # Колонка label → (N, 1)
    y_bin = df["label"].astype(int).values.reshape(-1, 1)

    # === Мультикласова ціль для класифікатора ===
    # Беремо колонку type, фіксуємо унікальні значення у порядку появи
    type_series = df["type"].astype(str)
    class_names = list(type_series.unique())  # порядок появи в датасеті
    type_to_idx = {name: i for i, name in enumerate(class_names)}
    y_idx = type_series.map(type_to_idx).values

    # Будуємо one-hot матрицю (N, C)
    n_samples = len(df)
    n_classes = len(class_names)
    y_mc = np.zeros((n_samples, n_classes), dtype=float)
    y_mc[np.arange(n_samples), y_idx] = 1.0

    # === Спліт на train/val (один і той самий для обох задач) ===
    # Спочатку розбиваємо X_all і бінарну мітку
    X_tr, y_bin_tr, X_va, y_bin_va = _train_val_split(
        X_all, y_bin, val_ratio=val_ratio, random_state=random_state
    )
    # Потім з тим самим random_state розбиваємо X_all і мультикласову мітку:
    # індекси будуть тими ж самими, тож X_tr/X_va збігатимуться для обох задач.
    _, y_mc_tr, _, y_mc_va = _train_val_split(
        X_all, y_mc, val_ratio=val_ratio, random_state=random_state
    )

    # === Функція масштабування (локальна) ===
    def _scale(X_train, X_val, mode: str):
        """
        Масштабує X_train, X_val і повертає також словник stats
        з параметрами масштабування для подальшого використання.
        """
        if mode == "zscore":
            # Z-score: (x - mean) / std
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True)
            std[std == 0] = 1.0  # захист від ділення на нуль
            X_train_s = (X_train - mean) / std
            X_val_s = (X_val - mean) / std
            return X_train_s, X_val_s, {"mean": mean, "std": std}
        elif mode == "minmax":
            # Min-Max: (x - xmin) / (xmax - xmin)
            xmin = X_train.min(axis=0, keepdims=True)
            xmax = X_train.max(axis=0, keepdims=True)
            denom = xmax - xmin
            denom[denom == 0] = 1.0  # щоб не ділити на 0
            X_train_s = (X_train - xmin) / denom
            X_val_s = (X_val - xmin) / denom
            return X_train_s, X_val_s, {"min": xmin, "max": xmax}
        else:
            # Без масштабування — просто повертаємо вхідні масиви
            return X_train, X_val, {}

    # === ДЕТЕКТОР (бінарна задача) ===
    # Масштабуємо копії X_tr, X_va (щоб не псувати X_all)
    X_det_tr, X_det_va, stats_det = _scale(
        X_tr.copy(), X_va.copy(), scaling_detector
    )
    y_det_tr, y_det_va = y_bin_tr, y_bin_va

    # За потреби балансуємо класи для тренування детектора
    if oversample_detector:
        X_det_tr, y_det_tr = _oversample_binary(
            X_det_tr, y_det_tr, random_state=random_state
        )

    # Метадані для детектора — все, що потрібно для відтворення preprocessing
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

    # === КЛАСИФІКАТОР (мультикласова задача) ===
    X_cls_tr, X_cls_va, stats_cls = _scale(
        X_tr.copy(), X_va.copy(), scaling_classifier
    )
    y_cls_tr, y_cls_va = y_mc_tr, y_mc_va

    # Oversampling рідкісних класів для класифікатора
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

    # === (ОПЦІЙНО) ЗБЕРЕЖЕННЯ У .NPZ ФАЙЛИ ===
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
