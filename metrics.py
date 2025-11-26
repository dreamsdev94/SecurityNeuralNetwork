import numpy as np


def _as_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)


# ---------- Бінарні метрики (мітки у {-1, +1}) ----------

def acc_sign(y_true, y_pred):
    """
    Accuracy за знаком виходу.
    y_true: {-1,+1}, y_pred: довільні дійсні.
    """
    yt = np.sign(_as_1d(y_true))
    yp = np.sign(_as_1d(y_pred))
    yp[yp == 0] = 1
    return float(np.mean(yt == yp))


def bin_f1_sign(y_true, y_pred):
    """
    F1-міра для бінарної класифікації за знаком.
    Позитивний клас = +1, негативний = -1.
    """
    yt = np.sign(_as_1d(y_true))
    yp = np.sign(_as_1d(y_pred))
    yp[yp == 0] = 1

    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == -1) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == -1))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)

    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(f1)


# ---------- Мультикласові метрики (мітки у {-1, +1} в one-vs-all) ----------

def argmax_classes(Y_pm1):
    """
    Перетворює матрицю у {-1,+1} (або довільні дійсні) в індекс класу за argmax.
    """
    Y_pm1 = np.asarray(Y_pm1)
    return np.argmax(Y_pm1, axis=1)


def accuracy_mc(Y_true_pm1, Y_pred_pm1):
    """
    Accuracy для мультикласового випадку.
    Y_true_pm1, Y_pred_pm1 – матриці форми (N, C) з оцінками класів.
    """
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    return float(np.mean(t == p))


def macro_f1_mc(Y_true_pm1, Y_pred_pm1):
    """
    Macro-F1 для мультикласу.
    """
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)

    classes = np.unique(t)
    f1s = []
    for c in classes:
        tp = np.sum((t == c) & (p == c))
        fp = np.sum((t != c) & (p == c))
        fn = np.sum((t == c) & (p != c))

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)

    return float(np.mean(f1s))


def topk_accuracy_mc(Y_true_pm1, Y_pred_pm1, k=3):
    """
    Top-k accuracy (наприклад, k=3).
    """
    Y_true_pm1 = np.asarray(Y_true_pm1)
    Y_pred_pm1 = np.asarray(Y_pred_pm1)

    t = argmax_classes(Y_true_pm1)

    topk = np.argsort(-Y_pred_pm1, axis=1)[:, :k]

    hits = 0
    for i in range(len(t)):
        if t[i] in topk[i]:
            hits += 1
    return float(hits / len(t))


def confusion_matrix_mc(Y_true_pm1, Y_pred_pm1):
    """
    Матриця невідповідностей (rows=true, cols=pred), тип int.
    """
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    C = int(Y_true_pm1.shape[1])
    M = np.zeros((C, C), dtype=int)
    for i in range(len(t)):
        M[t[i], p[i]] += 1
    return M
