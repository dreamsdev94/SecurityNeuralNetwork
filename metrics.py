import numpy as np


# ============================================================
# HELPERS
# ============================================================

def _argmax_classes(Y: np.ndarray) -> np.ndarray:
    """
    Приводить різні формати цільових / предиктованих значень до
    єдиного формату вектору цілочисельних індексів класів (N,).

    Підтримує:
      - one-hot матрицю (N, C) → індекси класів 0..C-1 (argmax по осі 1),
      - вектор (N,) з індексами → повертає як є (тільки приводить до int),
      - вектор (N,1)           → стискає до (N,) і приводить до int.

    Якщо форма не підходить — кидає ValueError.
    """
    Y = np.asarray(Y)

    # Випадок: уже вектор індексів класів (N,)
    if Y.ndim == 1:
        return Y.astype(int)

    # Випадок: матриця/вектор (N, 1) або (N, C)
    if Y.ndim == 2:
        if Y.shape[1] == 1:
            # (N,1) → (N,)
            return Y.reshape(-1).astype(int)
        # One-hot / ймовірності (N, C) → argmax по осі класів
        return np.argmax(Y, axis=1)

    # Будь-яка інша форма поки що не підтримується
    raise ValueError("Unsupported shape for class argmax")


# ============================================================
# METRICS FOR MULTICLASS PROBABILITIES
# ============================================================

def accuracy_mc(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Точність (accuracy) для мультикласової задачі.

    Y_true:
      - може бути у вигляді one-hot матриці (N, C),
      - або вектором індексів класів (N,),
      - або (N,1) з індексами.

    Y_pred:
      - матриця ймовірностей або логітів (N, C).
        Ми все одно беремо argmax, тож неважливо, чи це вже softmax,
        чи просто логіти.

    Обчислення:
      1) t = _argmax_classes(Y_true) → вектор істинних індексів,
      2) p = _argmax_classes(Y_pred) → вектор передбачених індексів,
      3) accuracy = mean(t == p).
    """
    t = _argmax_classes(Y_true)
    p = _argmax_classes(Y_pred)
    # Частка правильно класифікованих об’єктів
    return float(np.mean(t == p))


def macro_f1_mc(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Macro-F1 для мультикласової класифікації.

    Ідея Macro-F1:
      - рахуємо F1-міру окремо для кожного класу (як для "бінарної" задачі
        типу "клас c проти усіх інших"),
      - потім усереднюємо ці F1 по всіх класах (звичайне середнє арифм.).

    Це дає рівний вклад кожного класу незалежно від його частоти
    (на відміну від micro-F1 чи звичайної accuracy, де домінують часті класи).

    Формально для кожного класу c:
      tp_c — true positive (істинно-позитивні для класу c),
      fp_c — false positive (помилково зараховані до класу c),
      fn_c — false negative (клас c, передбачений не як c).

      precision_c = tp_c / (tp_c + fp_c)
      recall_c    = tp_c / (tp_c + fn_c)
      F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)

    Macro-F1 = mean(F1_c по всіх класах, де є хоча б один приклад).
    """
    # t — істинні індекси класів
    t = _argmax_classes(Y_true)
    # p — передбачені індекси класів
    p = _argmax_classes(Y_pred)

    # Кількість класів: припускаємо, що індекси йдуть від 0 до max(...)
    C = int(max(t.max(), p.max()) + 1)
    f1s = []  # список F1 для кожного класу

    for c in range(C):
        # Для класу c: позитивні — це ті, де t == c
        tp = np.sum((t == c) & (p == c))      # передбачили c і це було c
        fp = np.sum((t != c) & (p == c))      # передбачили c, але це не c
        fn = np.sum((t == c) & (p != c))      # це c, але передбачили не c

        # Якщо в даті взагалі немає прикладів класу c (ні t, ні p) —
        # пропускаємо його, щоб не псувати середнє "штучним нулем".
        if tp == 0 and fp == 0 and fn == 0:
            continue

        # Обережно рахуємо precision/recall, перевіряючи знаменники на 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Якщо і precision, і recall == 0, F1 визначаємо як 0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        f1s.append(f1)

    # Якщо з якоїсь причини жоден клас не потрапив у список (дуже патологічний випадок)
    if not f1s:
        return 0.0

    # Середнє F1 по всіх класах
    return float(np.mean(f1s))


def confusion_matrix_mc(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Матриця невідповідностей (confusion matrix) для мультикласу.

    Формат:
      - Рядки — істинні класи (true labels),
      - Стовпці — передбачені класи (predicted labels).

    M[i, j] показує, скільки об'єктів з істинним класом i
    було віднесено моделлю до класу j.

    Це корисно для детального аналізу помилок:
      - де модель плутає класи між собою,
      - які саме класи "змішуються".
    """
    # Приводимо до індексів класів
    t = _argmax_classes(Y_true)
    p = _argmax_classes(Y_pred)

    # Кількість класів — максимум з усіх індексів + 1
    C = int(max(t.max(), p.max()) + 1)

    # Створюємо порожню матрицю C x C, заповнену нулями (тип int)
    M = np.zeros((C, C), dtype=int)

    # Для кожного прикладу збільшуємо відповідний елемент
    # за правилами: рядок = true, стовпець = pred
    for i in range(len(t)):
        M[t[i], p[i]] += 1

    return M
