import numpy as np


# ============================================================
# HELPERS
# ============================================================

def _argmax_classes(Y: np.ndarray) -> np.ndarray:
    """
    Converts different formats of target / predicted values into
    a unified format: a vector of integer class indices (N,).

    Supports:
      - one-hot matrix (N, C) → class indices 0..C-1 (argmax along axis 1),
      - vector (N,) of indices → returns as is (only casts to int),
      - vector (N,1)           → squeezes to (N,) and casts to int.

    If the shape is not supported — raises ValueError.
    """
    Y = np.asarray(Y)

    # Case: already a vector of class indices (N,)
    if Y.ndim == 1:
        return Y.astype(int)

    # Case: matrix/vector (N, 1) or (N, C)
    if Y.ndim == 2:
        if Y.shape[1] == 1:
            # (N,1) → (N,)
            return Y.reshape(-1).astype(int)
        # One-hot / probabilities (N, C) → argmax over the class axis
        return np.argmax(Y, axis=1)

    # Any other shape is not supported yet
    raise ValueError("Unsupported shape for class argmax")


# ============================================================
# METRICS FOR MULTICLASS PROBABILITIES
# ============================================================

def accuracy_mc(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Accuracy for a multiclass task.

    Y_true:
      - can be a one-hot matrix (N, C),
      - or a vector of class indices (N,),
      - or (N,1) with indices.

    Y_pred:
      - a matrix of probabilities or logits (N, C).
        We take argmax anyway, so it does not matter whether this is already softmax
        or just raw logits.

    Computation:
      1) t = _argmax_classes(Y_true) → vector of true indices,
      2) p = _argmax_classes(Y_pred) → vector of predicted indices,
      3) accuracy = mean(t == p).
    """
    t = _argmax_classes(Y_true)
    p = _argmax_classes(Y_pred)
    # Fraction of correctly classified samples
    return float(np.mean(t == p))


def macro_f1_mc(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Macro-F1 for multiclass classification.

    Macro-F1 idea:
      - compute the F1-score separately for each class (as in a "binary" setup
        like "class c vs all others"),
      - then average these F1 scores across all classes (simple arithmetic mean).

    This gives equal contribution to each class regardless of its frequency
    (unlike micro-F1 or plain accuracy, where frequent classes dominate).

    Formally, for each class c:
      tp_c — true positives (true positives for class c),
      fp_c — false positives (incorrectly assigned to class c),
      fn_c — false negatives (class c predicted as not c).

      precision_c = tp_c / (tp_c + fp_c)
      recall_c    = tp_c / (tp_c + fn_c)
      F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)

    Macro-F1 = mean(F1_c over all classes that have at least one example).
    """
    # t — true class indices
    t = _argmax_classes(Y_true)
    # p — predicted class indices
    p = _argmax_classes(Y_pred)

    # Number of classes: assume indices go from 0 to max(...)
    C = int(max(t.max(), p.max()) + 1)
    f1s = []  # list of F1 scores for each class

    for c in range(C):
        # For class c: positives are those where t == c
        tp = np.sum((t == c) & (p == c))      # predicted c and it was c
        fp = np.sum((t != c) & (p == c))      # predicted c, but it was not c
        fn = np.sum((t == c) & (p != c))      # it was c, but predicted not c

        # If the dataset has no examples of class c at all (neither in t nor p) —
        # skip it so the mean is not distorted by an "artificial zero".
        if tp == 0 and fp == 0 and fn == 0:
            continue

        # Compute precision/recall carefully, checking denominators for 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # If both precision and recall are 0, define F1 as 0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        f1s.append(f1)

    # If for some reason no class ended up in the list (a very pathological case)
    if not f1s:
        return 0.0

    # Mean F1 across all classes
    return float(np.mean(f1s))


def confusion_matrix_mc(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix for multiclass classification.

    Format:
      - Rows — true classes (true labels),
      - Columns — predicted classes (predicted labels).

    M[i, j] shows how many samples with true class i
    were assigned by the model to class j.

    This is useful for detailed error analysis:
      - where the model confuses classes,
      - which classes get "mixed" together.
    """
    # Convert to class indices
    t = _argmax_classes(Y_true)
    p = _argmax_classes(Y_pred)

    # Number of classes — max index + 1
    C = int(max(t.max(), p.max()) + 1)

    # Create an empty C x C matrix filled with zeros (int type)
    M = np.zeros((C, C), dtype=int)

    # For each sample, increment the corresponding element
    # following the rule: row = true, column = pred
    for i in range(len(t)):
        M[t[i], p[i]] += 1

    return M
