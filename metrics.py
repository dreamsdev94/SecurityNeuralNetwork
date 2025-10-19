# metrics.py
import numpy as np

def _as_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)

# ---------- Binary (labels in {-1, +1}) ----------
def acc_sign(y_true, y_pred):
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    ps = np.sign(y_pred)
    ps[ps == 0] = 1
    return float(np.mean(ps == np.sign(y_true)))

def bin_confusion(y_true, y_pred):
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    ps = np.sign(y_pred); ps[ps == 0] = 1
    tp = int(np.sum((y_true == 1) & (ps == 1)))
    tn = int(np.sum((y_true == -1) & (ps == -1)))
    fp = int(np.sum((y_true == -1) & (ps == 1)))
    fn = int(np.sum((y_true == 1) & (ps == -1)))
    return tp, fp, fn, tn

def bin_prf(y_true, y_pred, eps=1e-12):
    tp, fp, fn, tn = bin_confusion(y_true, y_pred)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2*prec*rec / (prec+rec+eps)
    return prec, rec, f1

# ---------- Multiclass (one-vs-rest target in {-1,+1}, shape (N, C)) ----------
def argmax_classes(Y_pm1):
    return np.asarray(Y_pm1).argmax(axis=1)

def acc_argmax(Y_true_pm1, Y_pred_pm1):
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    return float(np.mean(t == p))

def top_k_acc(Y_true_pm1, Y_pred_pm1, k=3):
    t = argmax_classes(Y_true_pm1)
    pred = np.asarray(Y_pred_pm1)
    topk = np.argpartition(-pred, kth=k-1, axis=1)[:, :k]
    hit = np.any(topk == t[:, None], axis=1)
    return float(np.mean(hit))

def macro_f1(Y_true_pm1, Y_pred_pm1, eps=1e-12):
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    C = Y_true_pm1.shape[1]
    f1s = []
    for c in range(C):
        tp = np.sum((t == c) & (p == c))
        fp = np.sum((t != c) & (p == c))
        fn = np.sum((t == c) & (p != c))
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2*prec*rec / (prec+rec+eps)
        f1s.append(f1)
    return float(np.mean(f1s))

def confusion_matrix_mc(Y_true_pm1, Y_pred_pm1):
    t = argmax_classes(Y_true_pm1)
    p = argmax_classes(Y_pred_pm1)
    C = Y_true_pm1.shape[1]
    M = np.zeros((C, C), dtype=int)
    for i in range(len(t)):
        M[t[i], p[i]] += 1
    return M
