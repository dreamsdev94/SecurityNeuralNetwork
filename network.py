# network.py
import numpy as np

def xavier_init(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    W = rng.uniform(-limit, +limit, size=(fan_out, fan_in)).astype(np.float32)
    b = np.zeros((fan_out,), dtype=np.float32)
    return W, b

class Network:
    """
    Простий MLP з активацією tanh у прихованих шарах і tanh на виході
    (добре під ціль у {-1,+1} як для бінарного, так і для one-vs-rest).
    Оптимізатор: SGD або Adam.
    Втрати: MSE.
    """
    def __init__(self, layers, seed=42):
        """
        layers: [in, h1, ..., out]
        """
        self.layers = list(layers)
        self.rng = np.random.default_rng(seed)
        self.params = {}
        self.opt_state = {}
        self._init_params()

    def _init_params(self):
        self.params = {}
        L = len(self.layers) - 1
        for l in range(1, L+1):
            W, b = xavier_init(self.layers[l-1], self.layers[l], self.rng)
            self.params[f"W{l}"] = W
            self.params[f"b{l}"] = b
        # Adam buffers
        self.opt_state = {"m": {}, "v": {}, "t": 0}

    # -------- forward / backward --------
    @staticmethod
    def _tanh(x): return np.tanh(x)
    @staticmethod
    def _dtanh(y): return 1.0 - y*y  # y = tanh(x)

    def forward(self, X):
        """
        Повертає (activations, preactivations), де:
        A[0] = X
        для кожного шару l: Z[l] = Wl @ A[l-1]^T + b, Al = tanh(Z[l])
        """
        A = [X.astype(np.float32)]
        Z = [None]
        L = len(self.layers) - 1
        for l in range(1, L+1):
            W = self.params[f"W{l}"]; b = self.params[f"b{l}"]
            z = (A[-1] @ W.T) + b[None, :]
            a = self._tanh(z)
            Z.append(z); A.append(a)
        return A, Z

    def predict(self, X):
        A, _ = self.forward(X)
        return A[-1]  # tanh вихід у [-1,1]

    # -------- losses / grads --------
    @staticmethod
    def mse(Yhat, Y):
        diff = (Yhat - Y).astype(np.float32)
        return float(np.mean(diff*diff))

    def grads_mse(self, X, Y):
        A, Z = self.forward(X)
        Yhat = A[-1]
        N = X.shape[0]
        dA = (2.0 / N) * (Yhat - Y)  # dL/dA_L

        grads = {}
        L = len(self.layers) - 1
        for l in reversed(range(1, L+1)):
            dZ = dA * self._dtanh(A[l])
            dW = (dZ.T @ A[l-1])
            db = dZ.sum(axis=0)
            grads[f"dW{l}"] = dW.astype(np.float32)
            grads[f"db{l}"] = db.astype(np.float32)

            if l > 1:
                dA = dZ @ self.params[f"W{l}"]

        return grads

    # -------- optimizers --------
    def sgd_step(self, grads, lr):
        L = len(self.layers) - 1
        for l in range(1, L+1):
            self.params[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= lr * grads[f"db{l}"]

    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        st = self.opt_state
        st["t"] += 1
        t = st["t"]
        L = len(self.layers) - 1

        for l in range(1, L+1):
            for k, g in [(f"W{l}", grads[f"dW{l}"]), (f"b{l}", grads[f"db{l}"])]:
                if k not in st["m"]:
                    st["m"][k] = np.zeros_like(self.params[k])
                    st["v"][k] = np.zeros_like(self.params[k])

                st["m"][k] = beta1 * st["m"][k] + (1 - beta1) * g
                st["v"][k] = beta2 * st["v"][k] + (1 - beta2) * (g * g)

                m_hat = st["m"][k] / (1 - beta1**t)
                v_hat = st["v"][k] / (1 - beta2**t)

                self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # -------- batch generator --------
    @staticmethod
    def batch_iter(X, Y, batch_size=128, shuffle=True, rng=None):
        N = X.shape[0]
        idx = np.arange(N)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(idx)
        for i in range(0, N, batch_size):
            j = idx[i:i+batch_size]
            yield X[j], Y[j]

    # -------- training --------
    def fit(self, Xtr, Ytr, Xva, Yva,
            max_epochs=200, batch_size=128, lr=1e-3, l2=0.0,
            patience=10, min_delta=0.0, monitor="val_mse", mode=None,
            optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8,
            seed=42, warm_start=False, task_hint=None,
            early_stop_value=None, early_stop_rounds=2,
            reduce_lr_on_plateau=None, lr_patience=None, lr_min=1e-6,
            verbose_every=1,
            metrics_module=None):
        """
        monitor: "val_mse" | "val_acc" | "val_macroF1"
        mode: "min" або "max"; якщо None — підбирається під monitor
        task_hint: "binary" або "multiclass" — для друку метрик
        """
        rng = np.random.default_rng(seed)
        if not warm_start:
            self._init_params()  # ре-ініт ваг

        if metrics_module is None:
            raise ValueError("Pass metrics module with acc/f1 helpers")

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
        best_epoch = -1
        best_params = {k: v.copy() for k, v in self.params.items()}
        patience_ctr = 0
        no_improve = 0
        thr_ctr = 0
        curr_lr = float(lr)

        if reduce_lr_on_plateau is not None and lr_patience is None:
            lr_patience = max(5, patience // 2)

        def is_improved(score, best):
            if mode == "min":
                return (best - score) > min_delta
            else:
                return (score - best) > min_delta

        for epoch in range(max_epochs + 1):
            # --- train one epoch ---
            for Xb, Yb in self.batch_iter(Xtr, Ytr, batch_size=batch_size, shuffle=True, rng=rng):
                grads = self.grads_mse(Xb, Yb)

                # L2 (як weight decay)
                if l2 > 0.0:
                    Lnum = len(self.layers) - 1
                    for l in range(1, Lnum+1):
                        grads[f"dW{l}"] += l2 * self.params[f"W{l}"]

                if optimizer.lower() == "adam":
                    self.adam_step(grads, lr=curr_lr, beta1=beta1, beta2=beta2, eps=eps)
                else:
                    self.sgd_step(grads, lr=curr_lr)

            # --- eval ---
            Yhat_tr = self.predict(Xtr)
            Yhat_va = self.predict(Xva)

            train_mse = self.mse(Yhat_tr, Ytr)
            val_mse   = self.mse(Yhat_va, Yva)

            # metrics routing
            val_acc = val_prec = val_rec = val_f1 = np.nan
            val_macroF1 = val_top3 = np.nan

            if task_hint == "binary":
                val_acc = metrics_module.acc_sign(Yva, Yhat_va)
                val_prec, val_rec, val_f1 = metrics_module.bin_prf(Yva, Yhat_va)
            elif task_hint == "multiclass":
                val_acc = metrics_module.acc_argmax(Yva, Yhat_va)
                val_macroF1 = metrics_module.macro_f1(Yva, Yhat_va)
                val_top3 = metrics_module.top_k_acc(Yva, Yhat_va, k=3)

            # monitor score
            if monitor == "val_mse":
                score = val_mse
            elif monitor == "val_acc":
                score = val_acc
            elif monitor == "val_macroF1":
                score = val_macroF1
            else:
                raise ValueError(f"Unknown monitor {monitor}")

            # save history
            hist["epoch"].append(epoch)
            hist["train_mse"].append(train_mse)
            hist["val_mse"].append(val_mse)
            hist["val_acc"].append(val_acc)
            hist["val_prec"].append(val_prec)
            hist["val_rec"].append(val_rec)
            hist["val_f1"].append(val_f1)
            hist["val_macroF1"].append(val_macroF1)
            hist["val_top3"].append(val_top3)
            hist["lr"].append(curr_lr)

            if (epoch % verbose_every) == 0:
                if task_hint == "binary":
                    print(f"epoch {epoch:3d} | train_mse {train_mse:.8f} | val_mse {val_mse:.8f} | "
                          f"val_acc {100*val_acc:6.2f}% | P/R/F1 {val_prec:.3f}/{val_rec:.3f}/{val_f1:.3f}")
                else:
                    print(f"epoch {epoch:3d} | train_mse {train_mse:.8f} | val_mse {val_mse:.8f} | "
                          f"val_acc {100*val_acc:6.2f}% | macroF1 {val_macroF1:0.3f} | top3 {100*val_top3:6.2f}%")

            # check improvement
            if is_improved(score, best):
                best = score
                best_epoch = epoch
                best_params = {k: v.copy() for k, v in self.params.items()}
                patience_ctr = 0
                no_improve = 0
            else:
                patience_ctr += 1
                no_improve += 1

                # Reduce LR on plateau
                if (reduce_lr_on_plateau is not None) and (no_improve >= lr_patience):
                    new_lr = max(lr_min, curr_lr * float(reduce_lr_on_plateau))
                    if new_lr < curr_lr:
                        curr_lr = new_lr
                        print(f"[LR] No improvement for {no_improve} epochs → lr := {curr_lr:.2e}")
                    no_improve = 0

                # classical early stopping
                if patience_ctr >= patience:
                    print(f"Early stopping at epoch {epoch} (best {monitor} at epoch {best_epoch}: {best:.6f})")
                    self.params = best_params
                    break

            # threshold early stop
            if early_stop_value is not None:
                hit = (score >= early_stop_value) if mode == "max" else (score <= early_stop_value)
                thr_ctr = thr_ctr + 1 if hit else 0
                if thr_ctr >= (early_stop_rounds + 1):
                    print(f"Early stop on threshold: {monitor} reached {score:.6f} for {thr_ctr} epochs.")
                    self.params = best_params
                    break

        return hist

    # -------- save / load --------
    def save_weights(self, path):
        """Зберігає структуру та ваги в .npz"""
        flat = {"L": np.array(self.layers, dtype=np.int32)}
        L = len(self.layers) - 1
        for l in range(1, L+1):
            flat[f"W{l}"] = self.params[f"W{l}"]
            flat[f"b{l}"] = self.params[f"b{l}"]
        # Adam buffers збережемо опційно
        np.savez_compressed(path, **flat)

    @classmethod
    def load_weights(cls, path, seed=42):
        data = np.load(path, allow_pickle=True)
        layers = data["L"].tolist()
        net = cls(layers, seed=seed)
        L = len(layers) - 1
        for l in range(1, L+1):
            net.params[f"W{l}"] = data[f"W{l}"]
            net.params[f"b{l}"] = data[f"b{l}"]
        return net
