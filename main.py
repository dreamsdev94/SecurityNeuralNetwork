# mlp_sgd_vector.py
# -*- coding: utf-8 -*-
"""
Найпростіший векторизований MLP на NumPy:
- приховані шари з tanh
- вихідний шар за замовчуванням теж tanh (зручно під мітки {-1, 1})
- лосс: MSE
- оптимізатор: звичайний SGD (фіксований learning_rate), без Adam
- ініціалізація ваг: рівномірно в [-1, 1], bias = 0
- оцінка 'epoch 0' (до навчання)
- збереження/завантаження у JSON

Потрібен data_utils.create_dataset(...) -> (X_train, X_val, y_train, y_val),
де y зазвичай у {-1, 1}.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from data_utils import create_dataset

# ===================== 1) Конфіг моделі =====================

MODEL = {
    "hidden_layers": [],   # змінюй під себе (можна [] для логістичної регресії з tanh)
    "hidden_activation": "tanh",      # залишимо 'tanh' для простоти
    "output_dim": 1,
    "output_activation": "tanh",      # 'tanh' під мітки {-1,1}; (під {0,1} можеш поставити 'sigmoid')
    "seed": 123
}

# ===================== 2) Архітектура / ініціалізація =====================

"""
Побудова архітектури нейромережі: список розмірів шарів і відповідних активацій
input_size - кількість ознак
model - конфігурація нейромережі
"""
def build_arch(input_size, model):
    hidden = model["hidden_layers"]
    sizes = [int(input_size)] + list(map(int, hidden)) + [int(model["output_dim"])]
    activations = [model["hidden_activation"]] * len(hidden) + [model["output_activation"]]
    assert len(activations) == len(sizes) - 1
    return sizes, activations
"""
size - список кількості нейронів в шарах
activations - список функцій активації, які застосовуються до кожного шару
"""


"""
Ініціалізація ваг
size - список кількості нейронів в шарах
зерно ГВЧ для відтворюваності ініціалізації
"""
def init_uniform11(sizes, seed=42):
    """Ваги рівномірно в [-1, 1], bias = 0 — максимально просто."""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(sizes)):
        fan_in, fan_out = sizes[l-1], sizes[l]
        params[f"W{l}"] = rng.uniform(-1.0, 1.0, size=(fan_out, fan_in)).astype(np.float32)
        params[f"b{l}"] = np.zeros(fan_out, dtype=np.float32)
    return params
"""
params - параметри моделі
"""


# ===================== 3) Активації =====================
"""
values - тензор значень для активації
name - функція активації
"""
def act(values, name):
    name = name.lower()
    if name == "tanh":    return np.tanh(values)
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-values))
    if name == "linear":  return values
    raise ValueError(name)


"""
outputs - вихід активації
name - функція активації
"""
def act_deriv(outputs, name):
    """Похідна через ВИХІД a=act(z) — прості формули."""
    name = name.lower()
    if name == "tanh":    return 1.0 - outputs*outputs
    if name == "sigmoid": return outputs*(1.0 - outputs)
    if name == "linear":  return np.ones_like(outputs, dtype=outputs.dtype)
    raise ValueError(name)

# ===================== 4) Прямий / зворотний проходи =====================
"""
inputs - батч вхідних ознак
params - ваги й зсуви всіх шарів
activations - функції активації
"""
def forward(inputs, params, activations):
    """A[0]=X, A[-1]=вихід; все вектори/матриці."""
    outputs = [inputs.astype(np.float32)]
    before_act = []
    L = len(activations)
    for l in range(1, L+1):
        W, b = params[f"W{l}"], params[f"b{l}"]
        before_act_l = outputs[-1] @ W.T + b          # (N, in) @ (out, in)^T + (out,) -> (N, out)
        Al = act(before_act_l, activations[l-1])
        before_act.append(before_act_l); outputs.append(Al)
    return outputs, before_act
"""
outputs - список виходів кожного шару
before_act - список значень до активацій
"""

"""
output - передбачення моделі
target - цільове значення
"""
def mse_loss(output, target):
    diff = (output - target)
    return float(np.mean(diff*diff))

"""
outputs - список активацій з прямого проходу
targets - правильні значення для поточного батча
params - словник параметрів моделі - ваги та зсуви
activations - список функцій активації
l2=0.0 - коефіцієнт L2-регуляризації притягує ваги до нуля (зменшує перенавчання)
"""
def backward_mse(A, Y, params, activations, l2=0.0):
    """
    Проста MSE: dL/dA_L = 2/N * (A_L - Y)
    Далі стандартний backprop, похідні через act_deriv(A, name).
    """
    grads = {}
    N = max(1, Y.shape[0])
    L = len(activations)
    A_L = A[-1]
    # градієнт по виходу
    delta = (2.0 / N) * (A_L - Y) * act_deriv(A_L, activations[-1])

    for l in range(L, 0, -1):
        Wl = params[f"W{l}"]
        grads[f"W{l}"] = delta.T @ A[l-1] + l2 * Wl      # (out,N)@(N,in) -> (out,in)
        grads[f"b{l}"] = delta.sum(axis=0)               # (out,)
        if l > 1:
            delta = (delta @ Wl) * act_deriv(A[l-1], activations[l-2])
    return grads
"""
grads - словник градієнтів тієї ж структури що і params
"""


# ===================== 5) Звичайний SGD-крок =====================
"""
оновлення ваг та біасів за формулою
"""
def sgd_step(params, grads, learning_rate=1e-3):
    for k in params:
        params[k] -= learning_rate * grads[k]

# ===================== 6) Сервіси: батчі / метрики / I/O =====================
"""
Розбиття датасету на навчальні батчі
"""
def batch_iter(inputs, targets, batch_size=64, shuffle=True, seed=0):
    N = inputs.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, N, batch_size):
        sel = idx[i:i+batch_size]
        yield inputs[sel], targets[sel]

"""
Обчислення точності класифікації
outputs_pm1 - вихід моделі для батча прикладів
y_pm1 - правильні мітки для того ж батча
thr=0.0 - поріг для бінарізації
"""
def accuracy_tanh(outputs, targets, thr=0.0):
    """Точність за знаком для {-1,1}: threshold 0."""
    # Перетворення масиву значень в булевий масив, а потім в числа -1 або 1
    pred = (outputs >= thr).astype(np.float32)*2 - 1
    # приведення правильних міток в тип float
    y    = targets.astype(np.float32)
    return float(np.mean(pred == y))


def save_model_json(path, sizes, activations, params):
    payload = {"sizes": sizes, "activations": activations,
               "params": {k: v.tolist() for k, v in params.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_model_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = {k: np.array(v, dtype=np.float32) for k, v in payload["params"].items()}
    return payload["sizes"], payload["activations"], params

# ===================== 7) Тренування (SGD, фіксований LR, оцінка epoch 0) =====================
"""
Xtr: np.ndarray форми (N_tr, d) — тренувальні ознаки (float32 після касту).
Ytr_pm1: np.ndarray форми (N_tr, 1) — тренувальні мітки у {-1, 1}.
Xval: np.ndarray форми (N_val, d) — валідаційні ознаки.
Yval_pm1: np.ndarray форми (N_val, 1) — валідаційні мітки у {-1, 1}.
model_cfg: dict із полями типу:
"hidden_layers": список розмірів прихованих шарів, напр. [10,6] або [] для лише вихідного.
"hidden_activation": напр. "tanh".
"output_dim": розмір виходу (для бінарки — 1).
"output_activation": напр. "tanh".
"seed": int для відтворюваної ініціалізації.
max_epochs: int — кількість епох навчання.
batch_size: int — розмір мінібатча в batch_iter.
lr: float — фіксована швидкість навчання (learning rate) для SGD.
l2: float — коефіцієнт L2-регуляризації (0 — вимкнено).
model_out: str — шлях для збереження моделі (.json).
"""
def train_mlp_sgd(
    Xtr, Ytr_pm1, Xval, Yval_pm1,
    model_cfg,
    max_epochs=200, batch_size=64,
    lr=1e-4,
    l2=0.0,
    model_out="model_simple.json",
    # --- Early Stopping ---
    patience=10,             # скільки епох чекати без покращення
    min_delta=0.0,           # мінімальне покращення метрики, щоб "зарахувати" прогрес
    monitor="val_mse",       # яку метрику моніторити: "val_mse" або "val_acc"
    mode=None                # "min" для лоссів, "max" для метрик типу accuracy; якщо None — визначиться автоматично
):
    # Побудова архітектури
    sizes, activations = build_arch(Xtr.shape[1], model_cfg)

    # Каст типів
    Ytr = Ytr_pm1.astype(np.float32)
    Yval = Yval_pm1.astype(np.float32)
    Xtr = Xtr.astype(np.float32)
    Xval = Xval.astype(np.float32)

    # Параметри
    params = init_uniform11(sizes, seed=model_cfg.get("seed", 42))

    print("\n[ARCH: simplest SGD + EarlyStopping]")
    print(" sizes:", sizes)
    print(" activations:", activations)
    print(" optimizer: SGD (fixed lr)")
    print(" lr:", lr)

    history = {"epoch": [], "train_mse": [], "val_mse": [], "val_acc": []}

    # Початкова оцінка (epoch 0)
    Atr0, _ = forward(Xtr, params, activations)
    Av0,  _ = forward(Xval, params, activations)
    train0 = mse_loss(Atr0[-1], Ytr)
    val0   = mse_loss(Av0[-1], Yval)
    acc0   = accuracy_tanh(Av0[-1], Yval)

    print(f"epoch   0 | train_mse {train0:.5f} | val_mse {val0:.5f} | val_acc {acc0*100:6.2f}%")

    history["epoch"].append(0)
    history["train_mse"].append(train0)
    history["val_mse"].append(val0)
    history["val_acc"].append(acc0)

    # --- Early stopping setup ---
    if mode is None:
        mode = "min" if monitor.endswith("mse") or "loss" in monitor else "max"
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")

    best_score = float("inf") if mode == "min" else -float("inf")
    patience_counter = 0
    best_params = {k: v.copy() for k, v in params.items()}
    best_epoch = 0

    def is_improved(curr, best):
        if mode == "min":
            return (best - curr) > min_delta
        else:
            return (curr - best) > min_delta

    # Навчання
    for epoch in range(1, max_epochs + 1):
        for Xb, Yb in batch_iter(Xtr, Ytr, batch_size=batch_size, shuffle=True,
                                 seed=model_cfg.get("seed", 0) + epoch):
            A, _ = forward(Xb, params, activations)
            grads = backward_mse(A, Yb, params, activations, l2=l2)
            sgd_step(params, grads, learning_rate=lr)

        Atr, _ = forward(Xtr, params, activations)
        Av,  _ = forward(Xval, params, activations)

        train_mse = mse_loss(Atr[-1], Ytr)
        val_mse   = mse_loss(Av[-1], Yval)
        val_acc   = accuracy_tanh(Av[-1], Yval)
        print(f"epoch {epoch:3d} | train_mse {train_mse:.5f} | val_mse {val_mse:.5f} | val_acc {val_acc*100:6.2f}%")

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_acc"].append(val_acc)

        # --- Early stopping check ---
        curr_score = val_mse if monitor == "val_mse" else val_acc
        if is_improved(curr_score, best_score):
            best_score = curr_score
            patience_counter = 0
            best_params = {k: v.copy() for k, v in params.items()}
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best {monitor} at epoch {best_epoch}: {best_score:.6f})")
                # Повертаємо кращі ваги
                params = best_params
                break

    # Зберігаємо кращу модель
    save_model_json(model_out, sizes, activations, params)
    return params, history, (sizes, activations)
"""
params: dict з вагами та біасами ("W1","b1",...,"WL","bL"), np.float32.
history: dict з ключами epoch, train_mse, val_mse, val_acc — списки однакової довжини.
(sizes, activations): кортеж-архітектура (списки), щоб використовувати у predict(...)/збереженні.
"""
# ===================== 8) Інференс =====================
"""
Інференс - етап використання вже навченої моделі для отримання передбачень
"""
def predict(X, sizes, activations, params, mode="sign"):
    """Повертає або непороговані значення, або -1/1 за знаком."""
    A, _ = forward(X.astype(np.float32), params, activations)
    out = A[-1]
    if mode == "sign":
        return (out >= 0).astype(np.float32)*2 - 1
    return out

# ===================== 9) Запуск прикладу =====================

if __name__ == "__main__":
    # читаємо дані датасету з файлу
    Xtr, Xval, Ytr, Yval = create_dataset("Train_Test_Windows_10.csv")

    # Перетворення міток Y в NumPy-масив типу float32 (np.asarray(Ytr, dtype=np.float32))
    # Приведення форми до двовимірної колонки (N, 1)
    # -1 - підбір кількості рядків автоматично
    Ytr = np.asarray(Ytr, dtype=np.float32).reshape(-1, 1)
    Yval = np.asarray(Yval, dtype=np.float32).reshape(-1, 1)

    # тренування (SGD, фіксований lr)
    params, history, arch = train_mlp_sgd(
        Xtr, Ytr, Xval, Yval,
        MODEL,
        max_epochs=5000, batch_size=100,
        lr=1e-4, l2=0.0,
        model_out="model_simple.json",
        patience=10, min_delta=1e-5, monitor="val_mse", mode="min"
    )

    sizes, activations = arch

    # графік MSE
    plt.plot(history["epoch"], history["val_mse"], label="val_mse", marker="o")
    plt.plot(history["epoch"], history["train_mse"], label="train_mse", alpha=0.7)
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True); plt.legend()
    plt.title("Навчання MLP (SGD + MSE, tanh)")


    plt.savefig("training_mse.png", dpi=300, bbox_inches="tight")
    plt.close()

    # фінальна точність за знаком
    y_pred = predict(Xval, sizes, activations, params, mode="sign")
    final_acc = float(np.mean(y_pred == Yval))
    print(f"Final val_acc (sign): {final_acc*100:.2f}%")