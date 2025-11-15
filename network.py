import numpy as np

"""Функція повертає випадкові ваги та нульові зміщення за правилом Глорота (Xavier initialization):
fan_in — кількість входів у нейрон (кількість нейронів попереднього шару).
fan_out — кількість виходів (нейронів наступного шару).
rng — генератор випадкових чисел NumPy (np.random.default_rng())."""
def xavier_init(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out)) #Обчислення меж діапазону
    W = rng.uniform(-limit, +limit, size=(fan_out, fan_in)).astype(np.float32) #Ініціалізація ваг випадковими числами
    b = np.zeros((fan_out,), dtype=np.float32) #Ініціалізація вектору зміщень (нулі)
    return W, b


class Network:
    """
    MLP з підтримкою:
    - tanh (для бінарного завдання)
    - softmax (для мультикласової класифікації)
    - MSE або Cross-Entropy
    """

    def __init__(self, layers, seed=42, final_activation="tanh"):
        """
        layers: [in, h1, ..., out]
        final_activation: "tanh" або "softmax"
        """
        self.layers = list(layers)
        self.final_activation = final_activation.lower()

        assert self.final_activation in ["tanh", "softmax"], \
            "final_activation must be 'tanh' or 'softmax'"

        self.rng = np.random.default_rng(seed)
        self.params = {}
        self.opt_state = {}
        self._init_params()

    # ---------------------- INIT PARAMS ---------------------
    # Створює початкові ваги та зміщення для кожного шару за методом Xavier та внутрішні буфери для оптимізатора Adam.
    def _init_params(self):
        self.params = {}
        L = len(self.layers) - 1
        for l in range(1, L + 1):
            W, b = xavier_init(self.layers[l - 1], self.layers[l], self.rng)
            self.params[f"W{l}"] = W
            self.params[f"b{l}"] = b

        # Adam buffers
        self.opt_state = {"m": {}, "v": {}, "t": 0}

    # ------------------- ACTIVATION FUNCTIONS ----------------
    # Активаційна функція гіперболічний тангенс
    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    # Похідна тангенса
    @staticmethod
    def _dtanh(y):
        return 1.0 - y * y

    # Перетворює вихід останнього шару на ймовірності для мультикласової класифікації
    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)   # стабілізація
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # --------------------- LOSSES -----------------------
    # Обчислює середньоквадратичну помилку (MSE)
    @staticmethod
    def mse(Yhat, Y):
        diff = (Yhat - Y).astype(np.float32)
        return float(np.mean(diff * diff))

    # Обчислює крос-ентропію — стандартну помилку для мультикласової класифікації.
    @staticmethod
    def cross_entropy(Yhat, Y, eps=1e-12):
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return float(-np.mean(np.sum(Y * np.log(Yhat), axis=1)))

    # -------------------- FORWARD ------------------------
    # Прямий прохід мережі
    def forward(self, X):
        A = [X.astype(np.float32)]
        Z = [None]
        L = len(self.layers) - 1

        for l in range(1, L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            z = A[-1] @ W.T + b[None, :]
            Z.append(z)

            # останній шар
            if l == L:
                if self.final_activation == "softmax":
                    a = self._softmax(z)
                else:
                    a = self._tanh(z)
            else:
                a = self._tanh(z)

            A.append(a)
        return A, Z

    # -------------------- PREDICT ------------------------
    # Повертає вихід мережі для нових даних (завжди остання активація кінцевого шару)
    def predict(self, X):
        A, _ = self.forward(X)
        return A[-1]

    # ------------------ BACKPROP MSE ---------------------
    # Обчислює градієнти для MSE через backpropagation
    def grads_mse(self, X, Y):
        A, Z = self.forward(X)
        Yhat = A[-1]
        N = X.shape[0]

        dA = (2.0 / N) * (Yhat - Y)
        grads = {}

        L = len(self.layers) - 1
        for l in reversed(range(1, L + 1)):
            dZ = dA * (self._dtanh(A[l]) if l == L or self.final_activation == "tanh" else 1.0)
            dW = dZ.T @ A[l - 1]
            db = dZ.sum(axis=0)

            grads[f"dW{l}"] = dW.astype(np.float32)
            grads[f"db{l}"] = db.astype(np.float32)

            if l > 1:
                dA = dZ @ self.params[f"W{l}"]

        return grads

    # ------------------ BACKPROP CROSS-ENTROPY + SOFTMAX ---------------------
    # Обчислює градієнти для softmax + cross-entropy
    def grads_ce(self, X, Y):
        A, Z = self.forward(X)
        Yhat = A[-1]
        N = X.shape[0]

        # dZ = softmax - target
        dZ = (Yhat - Y) / N

        grads = {}
        L = len(self.layers) - 1

        for l in reversed(range(1, L + 1)):
            dW = dZ.T @ A[l - 1]
            db = dZ.sum(axis=0)

            grads[f"dW{l}"] = dW.astype(np.float32)
            grads[f"db{l}"] = db.astype(np.float32)

            # propagate to previous layer
            if l > 1:
                dA = dZ @ self.params[f"W{l}"]
                # hidden layers have tanh activation
                dZ = dA * self._dtanh(A[l - 1])

        return grads

    # ---------------------- OPTIMIZERS ----------------------
    # Оновлює ваги методом стохастичного градієнтного спуску (SGD)
    def sgd_step(self, grads, lr):
        L = len(self.layers) - 1
        for l in range(1, L + 1):
            self.params[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= lr * grads[f"db{l}"]

    # Оновлює ваги оптимізатором Adam (адаптивний метод з моментами)
    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        st = self.opt_state
        st["t"] += 1
        t = st["t"]

        L = len(self.layers) - 1

        for l in range(1, L + 1):
            for k in [f"W{l}", f"b{l}"]:
                g = grads[f"d{k}"] if f"d{k}" in grads else grads[f"d{k}"]

                if k not in st["m"]:
                    st["m"][k] = np.zeros_like(self.params[k])
                    st["v"][k] = np.zeros_like(self.params[k])

                st["m"][k] = beta1 * st["m"][k] + (1 - beta1) * g
                st["v"][k] = beta2 * st["v"][k] + (1 - beta2) * (g * g)

                m_hat = st["m"][k] / (1 - beta1**t)
                v_hat = st["v"][k] / (1 - beta2**t)

                self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------ BATCH GENERATOR ------------------------
    # Генерує міні-батчі для навчання
    @staticmethod
    def batch_iter(X, Y, batch_size=128, shuffle=True, rng=None):
        N = X.shape[0]
        idx = np.arange(N)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(idx)

        for i in range(0, N, batch_size):
            j = idx[i:i + batch_size]
            yield X[j], Y[j]

    # ---------------------------- FIT -------------------------------
    # Головний цикл навчання мережі.
    def fit(self, Xtr, Ytr, Xva, Yva,
            max_epochs=200, batch_size=128, lr=1e-3, l2=0.0,
            patience=10, min_delta=0.0, monitor="val_mse", mode=None,
            optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8,
            seed=42, warm_start=False, task_hint=None,
            early_stop_value=None, early_stop_rounds=2,
            reduce_lr_on_plateau=None, lr_patience=None, lr_min=1e-6,
            verbose_every=1,
            metrics_module=None):

        rng = np.random.default_rng(seed)
        if not warm_start:
            self._init_params()

        if metrics_module is None:
            raise ValueError("metrics_module is required")

        # історія
        hist = {
            "epoch": [], "train_mse": [], "val_mse": [],
            "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": [],
            "val_macroF1": [], "val_top3": [], "lr": []
        }

        # monitor mode
        if mode is None:
            mode = "min" if monitor == "val_mse" else "max"

        best = np.inf if mode == "min" else -np.inf
        best_params = {k: v.copy() for k, v in self.params.items()}
        patience_ctr = 0
        curr_lr = float(lr)

        # Training loop
        for epoch in range(max_epochs + 1):

            # --- training pass ---
            for Xb, Yb in self.batch_iter(Xtr, Ytr, batch_size=batch_size, shuffle=True, rng=rng):

                # вибір градієнта
                if self.final_activation == "softmax":
                    grads = self.grads_ce(Xb, Yb)
                else:
                    grads = self.grads_mse(Xb, Yb)

                # L2
                if l2 > 0.0:
                    Lnum = len(self.layers) - 1
                    for l in range(1, Lnum + 1):
                        grads[f"dW{l}"] += l2 * self.params[f"W{l}"]

                # apply optimizer
                if optimizer.lower() == "adam":
                    self.adam_step(grads, lr=curr_lr)
                else:
                    self.sgd_step(grads, lr=curr_lr)

            # --- evaluation ---
            Yhat_tr = self.predict(Xtr)
            Yhat_va = self.predict(Xva)

            if self.final_activation == "softmax":
                train_loss = self.cross_entropy(Yhat_tr, Ytr)
                val_loss = self.cross_entropy(Yhat_va, Yva)
            else:
                train_loss = self.mse(Yhat_tr, Ytr)
                val_loss = self.mse(Yhat_va, Yva)

            # метрики
            val_acc = val_prec = val_rec = val_f1 = np.nan
            val_macroF1 = val_top3 = np.nan

            if task_hint == "binary":
                val_acc = metrics_module.acc_sign(Yva, Yhat_va)
                val_prec, val_rec, val_f1 = metrics_module.bin_prf(Yva, Yhat_va)
            elif task_hint == "multiclass":
                val_acc = metrics_module.acc_argmax(Yva, Yhat_va)
                val_macroF1 = metrics_module.macro_f1(Yva, Yhat_va)
                val_top3 = metrics_module.top_k_acc(Yva, Yhat_va, k=3)

            # save history
            hist["epoch"].append(epoch)
            hist["train_mse"].append(train_loss)
            hist["val_mse"].append(val_loss)
            hist["val_acc"].append(val_acc)
            hist["val_prec"].append(val_prec)
            hist["val_rec"].append(val_rec)
            hist["val_f1"].append(val_f1)
            hist["val_macroF1"].append(val_macroF1)
            hist["val_top3"].append(val_top3)
            hist["lr"].append(curr_lr)

            # print
            if epoch % verbose_every == 0:
                print(f"epoch {epoch:3d} | loss_tr {train_loss:.6f} | loss_val {val_loss:.6f} | "
                      f"acc {val_acc}")

            # monitor improvement
            score = val_loss if monitor == "val_mse" else val_acc

            improved = (score < best - min_delta) if mode == "min" else (score > best + min_delta)

            if improved:
                best = score
                best_params = {k: v.copy() for k, v in self.params.items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

                if patience_ctr >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.params = best_params
                    break

        return hist

    # ---------------------------- SAVE / LOAD --------------------------------
    # Зберігає ваги та архітектуру мережі у .npz файл.
    def save_weights(self, path):
        flat = {"L": np.array(self.layers, dtype=np.int32)}
        L = len(self.layers) - 1
        for l in range(1, L + 1):
            flat[f"W{l}"] = self.params[f"W{l}"]
            flat[f"b{l}"] = self.params[f"b{l}"]
        np.savez_compressed(path, **flat)

    # Створює новий об’єкт мережі та завантажує в нього збережені ваги.
    @classmethod
    def load_weights(cls, path, seed=42):
        data = np.load(path, allow_pickle=True)
        layers = data["L"].tolist()
        net = cls(layers, seed=seed)
        L = len(layers) - 1
        for l in range(1, L + 1):
            net.params[f"W{l}"] = data[f"W{l}"]
            net.params[f"b{l}"] = data[f"b{l}"]
        return net
