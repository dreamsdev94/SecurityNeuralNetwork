import numpy as np


def accuracy(yhat, ytrue):
    """
    Загальна точність класифікації.

    Випадки:
    1) Мультиклас: матриці форми (N, C) – argmax по стовпцях.
    2) Бінарний: вектори або матриці (N,1).
       - якщо ytrue у {0,1} → порівнюємо напряму з порогом 0.5 для yhat;
       - інакше вважаємо, що це {-1,+1} і порівнюємо за знаком.
    """
    yhat = np.asarray(yhat)
    ytrue = np.asarray(ytrue)

    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)
    if ytrue.ndim == 1:
        ytrue = ytrue.reshape(-1, 1)

    # Мультиклас
    if yhat.shape[1] > 1 or ytrue.shape[1] > 1:
        preds = np.argmax(yhat, axis=1)
        true = np.argmax(ytrue, axis=1)
        return float(np.mean(preds == true))

    # Бінарний випадок
    preds = (yhat >= 0.5).astype(int).ravel()

    true_flat = ytrue.ravel()
    uniques = np.unique(true_flat)

    if np.all(np.isin(uniques, [0, 1])):
        true = true_flat.astype(int)
    else:
        # припускаємо {-1,+1}
        true = (true_flat >= 0).astype(int)

    return float(np.mean(preds == true))


def _as_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)


def acc_sign(yhat_pm1, ytrue_pm1):
    """
    Точність для бінарного випадку з мітками в {-1,+1}
    (порівнюємо знак).
    """
    yh = np.sign(_as_1d(yhat_pm1))
    yt = np.sign(_as_1d(ytrue_pm1))
    return float(np.mean(yh == yt))


def bin_f1_sign(yhat_pm1, ytrue_pm1):
    """
    F1-score для бінарного випадку з мітками в {-1,+1}.
    Повертає (precision, recall, f1).
    """
    yh = np.sign(_as_1d(yhat_pm1))
    yt = np.sign(_as_1d(ytrue_pm1))

    tp = np.sum((yh == 1) & (yt == 1))
    fp = np.sum((yh == 1) & (yt == -1))
    fn = np.sum((yh == -1) & (yt == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return float(precision), float(recall), float(f1)


def argmax_classes(Y_pm1):
    """
    Перетворює матрицю у {−1,+1} або в one-hot з будь-якими значеннями
    у прогноз класу через argmax по осі 1.
    """
    Y = np.asarray(Y_pm1)
    if Y.ndim == 1:
        return np.zeros_like(Y, dtype=int)
    return np.argmax(Y, axis=1)


def accuracy_mc(Y_pred_pm1, Y_true_pm1):
    """
    Точність мультикласової класифікації (через argmax).
    """
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    return float(np.mean(t == p))


def macro_f1_mc(Y_pred_pm1, Y_true_pm1):
    """
    Macro-F1 для мультикласової класифікації.
    """
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    classes = np.unique(t)
    f1_scores = []

    for c in classes:
        tp = np.sum((p == c) & (t == c))
        fp = np.sum((p == c) & (t != c))
        fn = np.sum((p != c) & (t == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    if len(f1_scores) == 0:
        return 0.0
    return float(np.mean(f1_scores))


def topk_accuracy_mc(Y_pred_pm1, Y_true_pm1, k=5):
    """
    Top-k accuracy для мультикласу.
    """
    Y_pred = np.asarray(Y_pred_pm1)
    Y_true = np.asarray(Y_true_pm1)

    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)

    true = np.argmax(Y_true, axis=1)
    topk = np.argsort(-Y_pred, axis=1)[:, :k]

    hits = 0
    for i in range(len(true)):
        if true[i] in topk[i]:
            hits += 1
    return float(hits / len(true))


def confusion_matrix_mc(Y_true_pm1, Y_pred_pm1):
    """
    Матриця невідповідностей (rows=true, cols=pred), тип int.
    """
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    C = int(np.max([t.max(), p.max()]) + 1)
    M = np.zeros((C, C), dtype=int)
    for i in range(len(t)):
        M[t[i], p[i]] += 1
    return M
