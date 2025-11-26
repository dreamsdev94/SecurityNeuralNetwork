# Імпорт бібліотеки Numpy
import numpy as np

try:
    # Спроба імпортувати бібліотеку CuPy
    import cupy as cp

    cupy_available = True
    try:
        cp.cuda.runtime.getDevice()
    except Exception:
        cupy_available = False
except Exception:
    cp = None
    cupy_available = False

# Використовуємо CuPy в разі доступності, інакше - Numpy
if cupy_available:
    xp = cp
else:
    xp = np


# а - будь який масив, np.float32 - необхідний тип даних
def to_xp(a, dtype=np.float32):
    if isinstance(a, xp.ndarray):
        return a
    return xp.asarray(a, dtype=dtype)


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "get"):
        return a.get()
    if hasattr(a, "asnumpy"):
        return a.asnumpy()
    return np.asarray(a)


# ======================================================================
#                           КЛАС НЕЙРОМЕРЕЖІ
# ======================================================================
class Network:

    def __init__(self, layers, seed=42):
        self.layers = list(layers)
        self.rng = np.random.default_rng(seed)
        self.params = {}
        self.opt_state = {}
        self._init_params()

    # ---------------- Xavier -----------------
    def _xavier_init(self, fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W_np = self.rng.uniform(-limit, +limit, size=(fan_out, fan_in)).astype(np.float32)
        b_np = np.zeros((fan_out,), dtype=np.float32)
        return to_xp(W_np), to_xp(b_np)

    # --------------- Ініціалізація параметрів ----------------
    def _init_params(self):
        self.params = {}
        L = len(self.layers) - 1

        for l in range(1, L + 1):
            W, b = self._xavier_init(self.layers[l - 1], self.layers[l])
            self.params[f"W{l}"] = W
            self.params[f"b{l}"] = b

        self.opt_state = {"m": {}, "v": {}, "t": 0}

    # =====================================================================
    #                          АКТИВАЦІЇ
    # =====================================================================
    @staticmethod
    def _tanh(x):
        return xp.tanh(x)

    @staticmethod
    def _dtanh(y):
        return 1.0 - y * y

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + xp.exp(-x))

    @staticmethod
    def _dsigmoid(y):
        return y * (1 - y)

    @staticmethod
    def _relu(x):
        return xp.maximum(0, x)

    @staticmethod
    def _drelu(x):
        return (x > 0).astype(xp.float32)

    @staticmethod
    def _softmax(x):
        x_shifted = x - xp.max(x, axis=1, keepdims=True)
        exp_x = xp.exp(x_shifted)
        return exp_x / xp.sum(exp_x, axis=1, keepdims=True)

    # ======================================================================
    #                        ПРЯМИЙ ПРОХІД
    # ======================================================================
    def forward(self, X, hidden_activation="tanh", output_activation="tanh"):
        X_xp = to_xp(X)

        A = [X_xp]
        Z = [None]

        L = len(self.layers) - 1

        for l in range(1, L + 1):

            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            z = A[-1] @ W.T + b[None, :]

            # приховані шари
            if l < L:
                if hidden_activation == "tanh":
                    a = self._tanh(z)
                elif hidden_activation == "relu":
                    a = self._relu(z)
                elif hidden_activation == "sigmoid":
                    a = self._sigmoid(z)
                else:
                    raise ValueError(f"Unknown activation: {hidden_activation}")

            # вихідний шар
            else:
                if output_activation == "tanh":
                    a = self._tanh(z)
                elif output_activation == "relu":
                    a = self._relu(z)
                elif output_activation == "sigmoid":
                    a = self._sigmoid(z)
                elif output_activation == "softmax":
                    a = self._softmax(z)
                else:
                    raise ValueError(f"Unknown output activation: {output_activation}")

            Z.append(z)
            A.append(a)

        return A, Z

    # -------------------------------------------------------------------------
    def predict(self, X, hidden_activation="tanh", output_activation="tanh"):
        A, _ = self.forward(X, hidden_activation, output_activation)
        return to_numpy(A[-1])

    # ======================================================================
    #                           ФУНКЦІЇ ВТРАТ
    # ======================================================================
    @staticmethod
    def mse(Yhat, Y):
        Yh = to_xp(Yhat)
        Yt = to_xp(Y)
        diff = Yh - Yt
        return float(xp.mean(diff * diff))

    @staticmethod
    def binary_cross_entropy(yhat, y):
        yhat = to_xp(yhat)
        y = to_xp(y)
        eps = 1e-12
        yhat = xp.clip(yhat, eps, 1 - eps)
        return float(-xp.mean(y * xp.log(yhat) + (1 - y) * xp.log(1 - yhat)))

    @staticmethod
    def cross_entropy(yhat, y):
        yhat = to_xp(yhat)
        y = to_xp(y)
        eps = 1e-12
        yhat = xp.clip(yhat, eps, 1 - eps)
        return float(-xp.sum(y * xp.log(yhat)) / y.shape[0])

    # ======================================================================
    #                         ЗВОРОТНЄ ПОШИРЕННЯ
    # ======================================================================
    def backward(self, X, Y, hidden_activation="tanh",
                 output_activation="tanh", loss="mse"):

        A, Z = self.forward(X, hidden_activation, output_activation)

        Y = to_xp(Y)
        Yhat = A[-1]
        N = Y.shape[0]

        # ----- dA від loss -----
        if loss == "mse":
            dA = (2.0 / N) * (Yhat - Y)

        elif loss == "bce":
            dA = (Yhat - Y) / N

        elif loss == "ce":
            dA = (Yhat - Y) / N

        else:
            raise ValueError(f"Unknown loss: {loss}")

        grads = {}
        L = len(self.layers) - 1

        # ---- backward ----
        for l in reversed(range(1, L + 1)):

            a_prev = A[l - 1]

            # вихідний шар
            if l == L:

                if (output_activation == "sigmoid" and loss == "bce") or \
                        (output_activation == "softmax" and loss == "ce"):
                    dZ = dA
                elif output_activation == "tanh":
                    dZ = dA * self._dtanh(Yhat)
                elif output_activation == "relu":
                    dZ = dA * self._drelu(Z[l])
                elif output_activation == "sigmoid":
                    dZ = dA * self._dsigmoid(Yhat)
                else:
                    raise NotImplementedError("Unsupported combination")

            # приховані шари
            else:
                if hidden_activation == "tanh":
                    dZ = dA * self._dtanh(A[l])
                elif hidden_activation == "relu":
                    dZ = dA * self._drelu(Z[l])
                elif hidden_activation == "sigmoid":
                    dZ = dA * self._dsigmoid(A[l])
                else:
                    raise ValueError("Unknown hidden activation")

            dW = dZ.T @ a_prev
            db = xp.sum(dZ, axis=0)

            grads[f"dW{l}"] = dW.astype(xp.float32)
            grads[f"db{l}"] = db.astype(xp.float32)

            if l > 1:
                dA = dZ @ self.params[f"W{l}"]

        return grads

    # ======================================================================
    #                           ОПТИМІЗАТОРИ
    # ======================================================================
    def sgd_step(self, grads, lr):
        L = len(self.layers) - 1
        for l in range(1, L + 1):
            self.params[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= lr * grads[f"db{l}"]

    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        st = self.opt_state
        st["t"] += 1
        t = st["t"]

        L = len(self.layers) - 1

        for l in range(1, L + 1):
            for k, g in [(f"W{l}", grads[f"dW{l}"]), (f"b{l}", grads[f"db{l}"])]:

                if k not in st["m"]:
                    st["m"][k] = xp.zeros_like(self.params[k])
                    st["v"][k] = xp.zeros_like(self.params[k])

                st["m"][k] = beta1 * st["m"][k] + (1 - beta1) * g
                st["v"][k] = beta2 * st["v"][k] + (1 - beta2) * (g * g)

                m_hat = st["m"][k] / (1 - beta1 ** t)
                v_hat = st["v"][k] / (1 - beta2 ** t)

                self.params[k] -= lr * m_hat / (xp.sqrt(v_hat) + eps)

    # ======================================================================
    #                      BATCH ITERATOR
    # ======================================================================
    @staticmethod
    def batch_iter(X, Y, batch_size=128, shuffle=True, rng=None):
        N = X.shape[0]
        idx = np.arange(N)

        if shuffle:
            rng = np.random.default_rng() if rng is None else rng
            rng.shuffle(idx)

        for i in range(0, N, batch_size):
            j = idx[i:i + batch_size]
            yield X[j], Y[j]

    # ======================================================================
    #                               FIT
    # ======================================================================
    def fit(self, Xtr, Ytr, Xva, Yva,
            max_epochs=200, batch_size=128, lr=1e-3,
            hidden_activation="tanh",
            output_activation="tanh",
            loss="mse",
            patience=10, min_delta=0.0,
            optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8,
            reduce_lr_on_plateau=None, lr_patience=None, lr_min=1e-6,
            verbose_every=1,
            seed=42, warm_start=False):

        rng = np.random.default_rng(seed)
        if not warm_start:
            self._init_params()

        n_out = self.layers[-1]
        n_hidden = len(self.layers) - 2

        # ------------------------------------------
        # Авто-режим
        # ------------------------------------------
        if loss == "auto":

            if n_out == 1:
                loss = "bce"
                hidden_activation = "sigmoid"
                output_activation = "sigmoid"
                print("[AUTO] Binary → sigmoid + BCE")

            elif n_out > 1:
                loss = "ce"
                hidden_activation = "tanh"
                output_activation = "softmax"
                print("[AUTO] Multiclass → tanh + softmax + CE")

        # >7 прихованих шарів → ReLU
        if n_hidden > 7:
            hidden_activation = "relu"
            print("[AUTO] >7 hidden layers → ReLU")

        # -----------------------------------------
        # Функція втрат
        # -----------------------------------------
        def compute_loss(Yhat, Y):
            if loss == "mse":
                return self.mse(Yhat, Y)
            elif loss == "bce":
                return self.binary_cross_entropy(Yhat, Y)
            elif loss == "ce":
                return self.cross_entropy(Yhat, Y)
            else:
                raise ValueError("Unknown loss")

        # -----------------------------------------
        # Логуємо
        # -----------------------------------------
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

        best = float("inf")
        best_epoch = -1
        best_params = {k: v.copy() for k, v in self.params.items()}
        patience_ctr = 0

        curr_lr = float(lr)
        no_improve = 0

        if reduce_lr_on_plateau is not None and lr_patience is None:
            lr_patience = max(5, patience // 2)

        def is_improved(score, best):
            return (best - score) > min_delta

        # =====================================================================
        #                        ТРЕНУВАННЯ
        # =====================================================================
        for epoch in range(max_epochs + 1):

            for Xb, Yb in self.batch_iter(Xtr, Ytr, batch_size=batch_size, shuffle=True, rng=rng):

                grads = self.backward(
                    Xb, Yb,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation,
                    loss=loss
                )

                if optimizer == "adam":
                    self.adam_step(grads, lr=curr_lr, beta1=beta1, beta2=beta2, eps=eps)
                else:
                    self.sgd_step(grads, lr=curr_lr)

            # ---- оцінка ----
            Yhat_tr = self.predict(Xtr, hidden_activation, output_activation)
            Yhat_va = self.predict(Xva, hidden_activation, output_activation)

            train_loss = compute_loss(Yhat_tr, Ytr)
            val_loss = compute_loss(Yhat_va, Yva)

            hist["epoch"].append(epoch)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["lr"].append(curr_lr)

            if verbose_every and epoch % verbose_every == 0:
                print(f"epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

            # --- Early stopping ---
            if is_improved(val_loss, best):
                best = val_loss
                best_epoch = epoch
                best_params = {k: v.copy() for k, v in self.params.items()}
                patience_ctr = 0
                no_improve = 0
            else:
                patience_ctr += 1
                no_improve += 1

                # Reduce LR
                if reduce_lr_on_plateau is not None and no_improve >= lr_patience:
                    new_lr = max(lr_min, curr_lr * reduce_lr_on_plateau)
                    if new_lr < curr_lr:
                        curr_lr = new_lr
                        print(f"[LR] ↓ {curr_lr:.2e}")
                    no_improve = 0

                # Early stop
                if patience_ctr >= patience:
                    print(f"[STOP] epoch {epoch} (best {best_epoch})")
                    self.params = best_params
                    break

        return hist

    # ======================================================================
    #                           SAVE / LOAD
    # ======================================================================
    def save_weights(self, path):
        flat = {"L": np.array(self.layers, dtype=np.int32)}
        L = len(self.layers) - 1

        for l in range(1, L + 1):
            flat[f"W{l}"] = to_numpy(self.params[f"W{l}"])
            flat[f"b{l}"] = to_numpy(self.params[f"b{l}"])

        np.savez_compressed(path, **flat)

    @classmethod
    def load_weights(cls, path, seed=42):
        data = np.load(path, allow_pickle=True)
        layers = data["L"].tolist()
        net = cls(layers, seed=seed)

        L = len(layers) - 1
        for l in range(1, L + 1):
            net.params[f"W{l}"] = to_xp(data[f"W{l}"])
            net.params[f"b{l}"] = to_xp(data[f"b{l}"])

        return net
