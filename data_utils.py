import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
#  ЗАГАЛЬНІ УТИЛІТИ ДЛЯ РОБОТИ З ДАНИМИ
# -------------------------------------------------------------------

def train_val_test_split(X, Y, val_size=0.2, test_size=0.2, random_state=42, stratify=True):
    """
    Розбиває дані на train/val/test.
    Якщо stratify=True, то стратифікуємо за мітками Y (за першою компонентою).
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    if stratify:
        # Стратифікація за першою компонентою
        y_strat = np.argmax(Y, axis=1) if Y.ndim == 2 and Y.shape[1] > 1 else (Y > 0).astype(int)
    else:
        y_strat = None

    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(
        X, Y, test_size=(val_size + test_size), random_state=random_state, stratify=y_strat
    )

    if stratify:
        y_tmp = np.argmax(Y_tmp, axis=1) if Y_tmp.ndim == 2 and Y_tmp.shape[1] > 1 else (Y_tmp > 0).astype(int)
    else:
        y_tmp = None

    rel_test = test_size / (val_size + test_size)

    X_va, X_te, Y_va, Y_te = train_test_split(
        X_tmp, Y_tmp, test_size=rel_test, random_state=random_state, stratify=y_tmp
    )

    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te


# -------------------------------------------------------------------
#  НОРМАЛІЗАЦІЯ ОЗНАК
# -------------------------------------------------------------------

def zscore_scale(X_tr, X_va=None, X_te=None):
    """
    Z-score нормалізація:
    x' = (x - mean) / std
    mean, std обчислюються лише за train.
    """
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)

    if X_va is not None:
        X_va_n = scaler.transform(X_va)
    else:
        X_va_n = None

    if X_te is not None:
        X_te_n = scaler.transform(X_te)
    else:
        X_te_n = None

    return X_tr_n, X_va_n, X_te_n, scaler


def minmax_0_1_scale(X_tr, X_va=None, X_te=None):
    """
    Нормалізація в [0, 1] (якщо раптом знадобиться):
    x' = (x - min) / (max - min)
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    xmin = X_tr.min(axis=0)
    xmax = X_tr.max(axis=0)
    denom = np.where((xmax - xmin) == 0, 1.0, (xmax - xmin))

    def _scale(X):
        if X is None:
            return None
        X = np.asarray(X, dtype=np.float32)
        return (X - xmin) / denom

    return _scale(X_tr), _scale(X_va), _scale(X_te), (xmin, xmax)


# -------------------------------------------------------------------
#  СТВОРЕННЯ НАБОРУ ДАНИХ
# -------------------------------------------------------------------

def create_dataset(
        X,
        Y,
        val_size=0.2,
        test_size=0.2,
        scaling="zscore",
        random_state=42,
        stratify=True
):
    """
    Повертає:
    Xtr, Ytr, Xva, Yva, Xte, Yte, info_dict

    scaling:
        "zscore"  – стандартна z-score нормалізація (рекомендовано)
        "0-1"     – нормалізація в [0, 1]
        None      – без масштабування
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    Xtr, Ytr, Xva, Yva, Xte, Yte = train_val_test_split(
        X, Y,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    scale_info = None

    if scaling == "zscore":
        Xtr_n, Xva_n, Xte_n, scaler = zscore_scale(Xtr, Xva, Xte)
        Xtr, Xva, Xte = Xtr_n, Xva_n, Xte_n
        scale_info = ("zscore", scaler)

    elif scaling == "0-1":
        Xtr_n, Xva_n, Xte_n, mm = minmax_0_1_scale(Xtr, Xva, Xte)
        Xtr, Xva, Xte = Xtr_n, Xva_n, Xte_n
        scale_info = ("0-1", mm)

    elif scaling is None:
        scale_info = None

    else:
        raise ValueError(f"Unknown scaling mode: {scaling}")

    info = {
        "val_size": val_size,
        "test_size": test_size,
        "scaling": scaling,
        "scale_info": scale_info,
        "random_state": random_state,
        "stratify": stratify
    }

    return Xtr, Ytr, Xva, Yva, Xte, Yte, info


# -------------------------------------------------------------------
#  ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ДАНИХ (ПРИКЛАД)
# -------------------------------------------------------------------

def make_synthetic_binary(n_samples=10000, n_features=113, random_state=42):
    """
    Простий приклад генерації бінарного датасету {-1, +1}.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    w_true = rng.normal(size=(n_features,)).astype(np.float32)
    scores = X @ w_true
    y = np.where(scores > 0, 1.0, -1.0).astype(np.float32)
    return X, y.reshape(-1, 1)


def make_synthetic_multiclass(n_samples=10000, n_features=113, n_classes=10, random_state=42):
    """
    Приклад генерації мультикласового датасету з мітками у {-1,+1} (one-vs-all).
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    W_true = rng.normal(size=(n_features, n_classes)).astype(np.float32)
    scores = X @ W_true

    labels = np.argmax(scores, axis=1)
    Y = -np.ones((n_samples, n_classes), dtype=np.float32)
    Y[np.arange(n_samples), labels] = 1.0
    return X, Y
