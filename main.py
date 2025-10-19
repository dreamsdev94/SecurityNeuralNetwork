# mlp_sgd_vector.py
# -*- coding: utf-8 -*-
"""
Найпростіший векторизований MLP на NumPy:
- приховані шари (за замовчуванням tanh)
- вихідний шар за замовчуванням теж tanh (зручно під мітки {-1, 1})
- лосс: MSE
- оптимізатор: SGD або Adam (з bias-correction)
- ініціалізація ваг: розумна (Xavier/He/LeCun) + bias = 0
- рання зупинка навчання (early stopping)
- оцінка 'epoch 0' (до навчання)
- збереження/завантаження у JSON
- збереження графіка навчання в PNG

Потрібен data_utils.create_dataset(...) -> (X_train, X_val, y_train, y_val),
де y зазвичай у {-1, 1}.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI (на сервері/headless)
import matplotlib.pyplot as plt
from data_utils import create_dataset

# ===================== 1) Конфіг моделі =====================

MODEL = {
    "hidden_layers": [],              # змінюй під себе (можна [] для "лінійної" моделі з tanh на виході)
    "hidden_activation": "tanh",      # залишимо 'tanh' для простоти
    "output_dim": 1,
    "output_activation": "tanh",      # 'tanh' під мітки {-1,1}; (під {0,1} можна 'sigmoid')
    "seed": 123
}

class Network:
    def __init__(self, model_config):
        self.model_config = model_config.copy()
        self.input_size = None
        self.optimizer = None
        self.sizes = None
        self.activations = None
        self.hidden = None
        self.arch = None
        self.params = None
        self.outputs = None
        self.before_act = None
        self.history = None

    def set_input_size(self, input_size):
        self.input_size = input_size

    def build_arch(self):

        if not self.input_size:
            raise ValueError("Input size must be defined by calling set_input_size()")

        self.hidden = self.model_config["hidden_layers"]
        self.sizes = [int(self.input_size)] + list(map(int, self.hidden)) + [int(self.model_config["output_dim"])]
        self.activations = [self.model_config["hidden_activation"]] * len(self.hidden) + [self.model_config["output_activation"]]
        assert len(self.activations) == len(self.sizes) - 1

        self.arch = {"sizes": self.sizes, "activations": self.activations}

    def init_smart(self, seed = 42):

        if not self.arch:
            raise ValueError("Architecture must be defined by calling build_arch()")

        random_gen = np.random.default_rng(seed)
        self.params = dict()

        def xavier_limit(fan_in, fan_out):
            return np.sqrt(6.0 / (fan_in + fan_out))

        for l in range(1, len(self.sizes)):
            fan_in, fan_out = self.sizes[l - 1], self.sizes[l]

            # Xavier (Glorot) для tanh/sigmoid/softmax/linear
            lim = xavier_limit(fan_in, fan_out)
            weight = random_gen.uniform(-lim, lim, size=(fan_out, fan_in))

            self.params[f"W{l}"] = weight.astype(np.float32)
            self.params[f"b{l}"] = np.zeros(fan_out, dtype=np.float32)

    @staticmethod
    def act(values, name):
        name = name.lower()
        if name == "tanh":    return np.tanh(values)
        if name == "sigmoid": return 1.0 / (1.0 + np.exp(-values))
        if name == "linear":  return values
        raise ValueError(name)

    @staticmethod
    def act_deriv(outputs, name):
        """Похідна через ВИХІД a=act(z)."""
        name = name.lower()
        if name == "tanh":    return 1.0 - outputs * outputs
        if name == "sigmoid": return outputs * (1.0 - outputs)
        if name == "linear":  return np.ones_like(outputs, dtype=outputs.dtype)
        raise ValueError(name)

    def forward(self, inputs):
        """A[0]=X, A[-1]=вихід; все вектори/матриці."""
        self.outputs = [inputs.astype(np.float32)]
        self.before_act = list()
        L = len(activations)
        for l in range(1, L + 1):
            weights, biases = params[f"W{l}"], params[f"b{l}"]
            before_act_l = self.outputs[-1] @ weights.T + biases  # (N, in) @ (out, in)^T + (out,) -> (N, out)
            output = act(before_act_l, activations[l - 1])
            self.before_act.append(before_act_l)
            self.outputs.append(output)

        return self.outputs

    @staticmethod
    def mse_loss(output, target):
        diff = (output - target)
        return float(np.mean(diff * diff))

    def backward_mse(self, targets, weight_decay=0.0):
        """
        MSE: dL/dA_L = 2/N * (A_L - Y)
        Далі стандартний backprop, похідні через act_deriv(A, name).
        """
        self.grads = dict()
        batch_size = max(1, targets.shape[0])
        layers_count = len(self.activations)
        final_output = self.outputs[-1]
        # градієнт по виходу
        delta = (2.0 / batch_size) * (final_output - targets) * act_deriv(final_output, self.activations[-1])

        for l in range(layers_count, 0, -1):
            weights_l = self.params[f"W{l}"]
            self.grads[f"W{l}"] = delta.T @ self.outputs[l - 1] + weight_decay * weights_l  # (out,N)@(N,in) -> (out,in)
            self.grads[f"b{l}"] = delta.sum(axis=0)  # (out,)
            if l > 1:
                delta = (delta @ weights_l) * act_deriv(self.outputs[l - 1], self.activations[l - 2])

    def sgd_step(self, learning_rate=1e-3):
        for k in self.params:
            params[k] -= learning_rate * self.grads[k]

    def adam_init_state(self):
        """Створює словники m і v під ті самі ключі, що й params."""
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        return {"m": m, "v": v, "t": 0}

    def adam_step(self, state, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam з bias-correction (m_hat, v_hat).
        В state['t'] рахуємо кроки ОНОВЛЕННЯ (зростає на кожний minibatch update).
        """
        state["t"] += 1
        t = state["t"]
        m, v = state["m"], state["v"]

        b1t = beta1 ** t
        b2t = beta2 ** t
        corr1 = 1.0 - b1t
        corr2 = 1.0 - b2t

        for k in params:
            grad = self.grads[k]
            m[k] = beta1 * m[k] + (1.0 - beta1) * grad
            v[k] = beta2 * v[k] + (1.0 - beta2) * (grad * grad)
            m_hat = m[k] / max(corr1, 1e-16)
            v_hat = v[k] / max(corr2, 1e-16)
            self.params[k] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    @staticmethod
    def batch_iter(inputs, targets, batch_size=64, shuffle=True, seed=0):
        n_samples = inputs.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            sel = indices[i:i + batch_size]
            yield inputs[sel], targets[sel]

    @staticmethod
    def accuracy_tanh(outputs, targets, thr=0.0):
        """Точність за знаком для {-1,1}: threshold 0."""
        pred = (outputs >= thr).astype(np.float32) * 2 - 1
        y = targets.astype(np.float32)
        return float(np.mean(pred == y))

    def save_model_json(self, path):
        payload = {"sizes": self.sizes, "activations": self.activations,
                   "params": {k: v.tolist() for k, v in self.params.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_model_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.params = {k: np.array(v, dtype=np.float32) for k, v in payload["params"].items()}

    def train_mlp_sgd(self,
            train_sets, train_targets, valid_sets, valid_targets,
            max_epochs=200, batch_size=64,
            learning_rate=1e-3,
            l2=0.0,
            model_out="model_simple.json",
            # --- Early Stopping ---
            patience=10,
            min_delta=0.0,
            early_stop_mode=None,
            # --- Optimizer selection ---
            optimizer="sgd",  # "sgd" або "adam"
            beta1=0.9, beta2=0.999, eps=1e-8
    ):
        # Архітектура
        self.build_arch()

        # Касти типів
        train_targets = train_targets.astype(np.float32)
        valid_targets = valid_targets.astype(np.float32)
        train_sets = train_sets.astype(np.float32)
        valid_sets = valid_sets.astype(np.float32)

        # Ініціалізація параметрів (SMART)
        self.init_smart(seed=self.model_config.get("seed", 42))
        # (опційно) трохи зменшити масштаб вихідного шару з tanh, щоб менше насичуватись на старті
        if self.activations[-1].lower() == "tanh":
            self.params[f"W{len(sizes) - 1}"] *= 0.5

        # Ініціалізація стану оптимізатора (для Adam)
        opt_state = None
        if optimizer.lower() == "adam":
            opt_state = adam_init_state(self.params)

        print("\n[ARCH: simplest SGD/Adam + EarlyStopping + SmartInit]")
        print(" sizes:", self.sizes)
        print(" activations:", self.activations)
        print(f" optimizer: {optimizer.upper()}")
        print(" lr:", learning_rate)
        if optimizer.lower() == "adam":
            print(f" adam(beta1={beta1}, beta2={beta2}, eps={eps})")

        self.history = {"epoch": [], "train_mse": [], "val_mse": [], "val_acc": []}

        # Початкова оцінка (epoch 0)

        zero_train_activations, _ = forward(train_sets, self.params, self.activations)
        zero_valid_activations, _ = forward(valid_sets, self.params, self.activations)
        zero_train_mse = mse_loss(zero_train_activations[-1], train_targets)
        zero_valid_mse = mse_loss(zero_valid_activations[-1], Yval)
        zero_accuracy = accuracy_tanh(zero_valid_activations[-1], Yval)

        print(f"epoch   0 | train_mse {zero_train_mse:.5f} | val_mse {zero_valid_mse:.5f} | val_acc {zero_accuracy * 100:6.2f}%")

        self.history["epoch"].append(0)
        self.history["train_mse"].append(zero_train_mse)
        self.history["val_mse"].append(zero_valid_mse)
        self.history["val_acc"].append(zero_accuracy)

        # --- Early stopping setup ---
        if early_stop_mode is None:
            mode = "min"
        if early_stop_mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        best_score = float("inf") if early_stop_mode == "min" else -float("inf")
        patience_counter = 0
        best_params = {k: v.copy() for k, v in self.params.items()}
        best_epoch = 0

        def is_improved(curr, best):
                return (best - curr) > min_delta


        # Навчання
        for epoch in range(1, max_epochs + 1):
            # епоха проходів по мінібатчах
            for Xb, Yb in batch_iter(train_sets, train_targets, batch_size=batch_size, shuffle=True,
                                     seed=self.model_config.get("seed", 0) + epoch):
                A, _ = forward(Xb, self.params, activations)
                grads = backward_mse(A, Yb, self.params, activations, l2=l2)

                if optimizer.lower() == "adam":
                    adam_step(self.params, grads, opt_state, lr=learning_rate, beta1=beta1, beta2=beta2, eps=eps)
                else:
                    sgd_step(self.params, grads, learning_rate=learning_rate)

            # оцінка після епохи
            train_activations, _ = forward(train_sets, self.params, activations)
            valid_activations, _ = forward(Xval, self.params, activations)

            train_mse = mse_loss(train_activations[-1], train_targets)
            val_mse = mse_loss(valid_activations[-1], valid_targets)
            val_acc = accuracy_tanh(valid_activations[-1], valid_targets)
            print(
                f"epoch {epoch:3d} | train_mse {train_mse:.5f} | val_mse {val_mse:.5f} | val_acc {val_acc * 100:6.2f}%")

            self.history = history["epoch"].append(epoch)
            self.history = history["train_mse"].append(train_mse)
            self.history = history["val_mse"].append(val_mse)
            self.history = history["val_acc"].append(val_acc)

            # --- Early stopping check ---
            curr_score = val_mse
            if is_improved(curr_score, best_score):
                best_score = curr_score
                patience_counter = 0
                best_params = {k: v.copy() for k, v in self.params.items()}
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (best val_mse at epoch {best_epoch}: {best_score:.6f})")
                    # Повертаємо кращі ваги
                    self.params = best_params
                    break

        # Зберігаємо кращу модель
        save_model_json(model_out, self.sizes, self.activations, self.params)

    def predict(self, input):
        """Повертає або непороговані значення, або -1/1 за знаком."""
        output = self.forward(input.astype(np.float32))
        result = list()
        if self.activations[-1] == "tanh":

            for i in range(len(output)):
                elem = 1 if output[i] >= 0.0 else -1
                result.append(elem)



        if self.activations[-1] == "sigmoid": pass
        if self.activations[-1] == "linear":  pass

        return result




# ===================== 2) Архітектура / ініціалізація =====================

def build_arch(input_size, model):
    """Повертає розміри шарів (sizes) і список активацій для кожного шару (activations)."""
    hidden = model["hidden_layers"]
    sizes = [int(input_size)] + list(map(int, hidden)) + [int(model["output_dim"])]
    activations = [model["hidden_activation"]] * len(hidden) + [model["output_activation"]]
    assert len(activations) == len(sizes) - 1
    return sizes, activations
def init_uniform11(sizes, seed=42):
    """(Стара базова) Ваги рівномірно в [-1, 1], bias = 0."""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(sizes)):
        fan_in, fan_out = sizes[l-1], sizes[l]
        params[f"W{l}"] = rng.uniform(-1.0, 1.0, size=(fan_out, fan_in)).astype(np.float32)
        params[f"b{l}"] = np.zeros(fan_out, dtype=np.float32)
    return params
def init_smart(sizes, activations, seed=42, dist="auto"):
    """
    Ініціалізує ваги згідно з активацією кожного шару:
    - tanh/sigmoid/softmax/linear -> Xavier (Glorot)
    - ReLU/LeakyReLU/GELU/SiLU(Swish)/ELU -> He (Kaiming)
    - SELU -> LeCun normal
    bias = 0
    """
    rng = np.random.default_rng(seed)
    params = {}

    def he_limit(fan_in):    # для рівномірного варіанта He
        return np.sqrt(6.0 / fan_in)

    def xavier_limit(fan_in, fan_out):
        return np.sqrt(6.0 / (fan_in + fan_out))

    for l in range(1, len(sizes)):
        fan_in, fan_out = sizes[l-1], sizes[l]
        act = activations[l-1].lower()


        # Xavier (Glorot) для tanh/sigmoid/softmax/linear
        lim = xavier_limit(fan_in, fan_out)
        W = rng.uniform(-lim, lim, size=(fan_out, fan_in))

        params[f"W{l}"] = W.astype(np.float32)
        params[f"b{l}"] = np.zeros(fan_out, dtype=np.float32)

    return params
def act(values, name):
    name = name.lower()
    if name == "tanh":    return np.tanh(values)
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-values))
    if name == "linear":  return values
    raise ValueError(name)
def act_deriv(outputs, name):
    """Похідна через ВИХІД a=act(z)."""
    name = name.lower()
    if name == "tanh":    return 1.0 - outputs*outputs
    if name == "sigmoid": return outputs*(1.0 - outputs)
    if name == "linear":  return np.ones_like(outputs, dtype=outputs.dtype)
    raise ValueError(name)
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
def mse_loss(output, target):
    diff = (output - target)
    return float(np.mean(diff*diff))
def backward_mse(A, Y, params, activations, l2=0.0):
    """
    MSE: dL/dA_L = 2/N * (A_L - Y)
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
def sgd_step(params, grads, learning_rate=1e-3):
    for k in params:
        params[k] -= learning_rate * grads[k]
def adam_init_state(params):
    """Створює словники m і v під ті самі ключі, що й params."""
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    return {"m": m, "v": v, "t": 0}
def adam_step(params, grads, state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam з bias-correction (m_hat, v_hat).
    В state['t'] рахуємо кроки ОНОВЛЕННЯ (зростає на кожний minibatch update).
    """
    state["t"] += 1
    t = state["t"]
    m, v = state["m"], state["v"]

    b1t = beta1 ** t
    b2t = beta2 ** t
    corr1 = 1.0 - b1t
    corr2 = 1.0 - b2t

    for k in params:
        g = grads[k]
        m[k] = beta1 * m[k] + (1.0 - beta1) * g
        v[k] = beta2 * v[k] + (1.0 - beta2) * (g * g)
        m_hat = m[k] / max(corr1, 1e-16)
        v_hat = v[k] / max(corr2, 1e-16)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
def batch_iter(inputs, targets, batch_size=64, shuffle=True, seed=0):
    N = inputs.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, N, batch_size):
        sel = idx[i:i+batch_size]
        yield inputs[sel], targets[sel]
def accuracy_tanh(outputs, targets, thr=0.0):
    """Точність за знаком для {-1,1}: threshold 0."""
    pred = (outputs >= thr).astype(np.float32)*2 - 1
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

def train_mlp_sgd(
    Xtr, Ytr_pm1, Xval, Yval_pm1,
    model_cfg,
    max_epochs=200, batch_size=64,
    lr=1e-3,
    l2=0.0,
    model_out="model_simple.json",
    # --- Early Stopping ---
    patience=10,
    min_delta=0.0,
    monitor="val_mse",
    mode=None,
    # --- Optimizer selection ---
    optimizer="sgd",       # "sgd" або "adam"
    beta1=0.9, beta2=0.999, eps=1e-8
):
    # Архітектура
    sizes, activations = build_arch(Xtr.shape[1], model_cfg)

    # Касти типів
    Ytr  = Ytr_pm1.astype(np.float32)
    Yval = Yval_pm1.astype(np.float32)
    Xtr  = Xtr.astype(np.float32)
    Xval = Xval.astype(np.float32)

    # Ініціалізація параметрів (SMART)
    params = init_smart(sizes, activations, seed=model_cfg.get("seed", 42))
    # (опційно) трохи зменшити масштаб вихідного шару з tanh, щоб менше насичуватись на старті
    if activations[-1].lower() == "tanh":
        params[f"W{len(sizes)-1}"] *= 0.5

    # Ініціалізація стану оптимізатора (для Adam)
    opt_state = None
    if optimizer.lower() == "adam":
        opt_state = adam_init_state(params)

    print("\n[ARCH: simplest SGD/Adam + EarlyStopping + SmartInit]")
    print(" sizes:", sizes)
    print(" activations:", activations)
    print(f" optimizer: {optimizer.upper()}")
    print(" lr:", lr)
    if optimizer.lower() == "adam":
        print(f" adam(beta1={beta1}, beta2={beta2}, eps={eps})")

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
        # епоха проходів по мінібатчах
        for Xb, Yb in batch_iter(Xtr, Ytr, batch_size=batch_size, shuffle=True,
                                 seed=model_cfg.get("seed", 0) + epoch):
            A, _ = forward(Xb, params, activations)
            grads = backward_mse(A, Yb, params, activations, l2=l2)

            if optimizer.lower() == "adam":
                adam_step(params, grads, opt_state, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
            else:
                sgd_step(params, grads, learning_rate=lr)

        # оцінка після епохи
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

# ===================== 8) Інференс =====================

def predict(self, input):
    """Повертає або непороговані значення, або -1/1 за знаком."""
    output, _ = self.forward(input.astype(np.float32))

    if name == "tanh":    return np.tanh(values)
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-values))
    if name == "linear":  return values

    out = A[-1]
    if mode == "sign":
        return (out >= 0).astype(np.float32)*2 - 1
    return out

# ===================== 9) Запуск прикладу =====================

if __name__ == "__main__":
    # читаємо дані датасету з файлу
    Xtr, Xval, Ytr, Yval = create_dataset("Train_Test_Windows_10.csv")

    # Перетворення міток Y в NumPy-масив типу float32 та форма (N,1)
    Ytr  = np.asarray(Ytr,  dtype=np.float32).reshape(-1, 1)
    Yval = np.asarray(Yval, dtype=np.float32).reshape(-1, 1)

    # тренування (тепер можна вибрати оптимізатор)
    params, history, arch = train_mlp_sgd(
        Xtr, Ytr, Xval, Yval,
        MODEL,
        max_epochs=5000, batch_size=100,
        lr=1e-3,              # для Adam зазвичай більше, ніж для SGD з того ж масштабу
        l2=0.0,
        model_out="model_simple.json",
        patience=10, min_delta=1e-5, monitor="val_mse", mode="min",
        optimizer="adam",      # <-- вибір: "adam" або "sgd"
        beta1=0.9, beta2=0.999, eps=1e-8
    )

    sizes, activations = arch

    # графік MSE (зберегти у файл, без показу)
    plt.figure()
    plt.plot(history["epoch"], history["val_mse"], label="val_mse", marker="o")
    plt.plot(history["epoch"], history["train_mse"], label="train_mse", alpha=0.7)
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True); plt.legend()
    plt.title("Навчання MLP (SGD/Adam + MSE, tanh)")
    plt.tight_layout()
    plt.savefig("training_mse.png", dpi=300, bbox_inches="tight")
    plt.close()

    # фінальна точність за знаком
    y_pred = predict(Xval, sizes, activations, params, mode="sign")
    final_acc = float(np.mean(y_pred == Yval))
    print(f"Final val_acc (sign): {final_acc*100:.2f}%")