import os
import numpy as np
from metrics import accuracy_mc, macro_f1_mc

# ============================================================
# GPU / CPU backend with runtime switch
# ============================================================

# By default — CPU; in main.py we call set_backend(...)
import numpy as xp  # type: ignore
BACKEND = "CPU (NumPy)"
USE_GPU = False
GPU_AVAILABLE = False


def set_backend(use_gpu: bool = False):
    """
    Switch between NumPy (CPU) and CuPy (GPU).

    Usage:
        from network import set_backend
        set_backend(True)   # try GPU (CuPy)
        set_backend(False)  # force CPU

    If the GPU is unavailable or an error occurs when importing cupy, the network
    automatically falls back to NumPy (CPU).
    """
    global xp, BACKEND, USE_GPU, GPU_AVAILABLE

    USE_GPU = bool(use_gpu)

    if USE_GPU:
        try:
            import cupy as _cp  # type: ignore
            xp = _cp
            BACKEND = "GPU (CuPy)"
            GPU_AVAILABLE = True
        except Exception:
            # In case of any error — fallback to NumPy
            import numpy as _np  # type: ignore
            xp = _np
            BACKEND = "CPU (NumPy)"
            GPU_AVAILABLE = False
    else:
        import numpy as _np  # type: ignore
        xp = _np
        BACKEND = "CPU (NumPy)"
        GPU_AVAILABLE = False

    print(f"[BACKEND] Using: {BACKEND}")


# ============================================================
# Network
# ============================================================

class Network:
    """
    A multilayer fully-connected neural network with support for:
      - different weight initializations (Xavier/He/uniform);
      - activations (ReLU, tanh, sigmoid, GELU, softmax);
      - LayerNorm on hidden layers;
      - loss functions (BCE, CE, MSE);
      - Adam optimizer;
      - early stopping and metric logging;
      - saving/loading the model to/from .npz.
    """

    # --------------------- INITIALIZATIONS ---------------------

    @staticmethod
    def init_weights(fan_in, fan_out, method: str = "xavier_normal"):
        """
        Weight initialization for a layer of size fan_in x fan_out.

        method:
          - "xavier_uniform" : U(-a, a), where a = sqrt(6 / (fan_in + fan_out))
          - "xavier_normal"  : N(0, 2 / (fan_in + fan_out))
          - "he"             : N(0, 2 / fan_in)
          - other            : uniform distribution [-1, 1]
        """
        if method == "xavier_uniform":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
        elif method == "xavier_normal":
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            w = np.random.randn(fan_in, fan_out) * scale
        elif method == "he":
            scale = np.sqrt(2.0 / fan_in)
            w = np.random.randn(fan_in, fan_out) * scale
        else:
            # random initialization in the range [-1, 1]
            w = xp.random.uniform(-1.0, 1.0, (fan_in, fan_out))

        # Convert the array to the current backend (xp = np or cupy)
        return xp.asarray(w, dtype=xp.float32)

    # ------------------------ ACTIVATIONS ------------------------

    @staticmethod
    def relu(x):
        # ReLU(x) = max(0, x)
        return xp.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        # d/dx ReLU(x): 1 for x>0, 0 otherwise
        return (x > 0).astype(xp.float32)

    @staticmethod
    def tanh(x):
        return xp.tanh(x)

    @staticmethod
    def tanh_deriv(a):
        # a = tanh(x), so the derivative = 1 - a^2
        return 1 - a * a

    @staticmethod
    def sigmoid(x):
        # sigmoid(x) = 1 / (1 + e^{-x})
        return 1 / (1 + xp.exp(-x))

    @staticmethod
    def sigmoid_deriv(a):
        # a = sigmoid(x), so the derivative = a * (1 - a)
        return a * (1 - a)

    @staticmethod
    def gelu(x):
        """
        GELU activation (approximation):
        0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        """
        # use np for constants, xp for tensors
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1 + xp.tanh(c * (x + 0.044715 * x ** 3)))

    @staticmethod
    def gelu_deriv(x):
        """
        Derivative of GELU (for backprop).
        The formula is implemented by differentiating the approximation above.
        """
        c = np.sqrt(2.0 / np.pi)
        t = xp.tanh(c * (x + 0.044715 * x ** 3))
        dt = (1 - t ** 2) * c * (1 + 3 * 0.044715 * x ** 2)
        return 0.5 * (1 + t) + 0.5 * x * dt

    @staticmethod
    def softmax(x):
        """
        Softmax over the last dimension (axis=1) with a prior shift
        (subtracting the maximum) for numerical stability.
        """
        x = x - xp.max(x, axis=1, keepdims=True)
        e = xp.exp(x)
        return e / xp.sum(e, axis=1, keepdims=True)

    # ------------------------ LayerNorm -------------------------

    @staticmethod
    def layer_norm_forward(Z, eps=1e-5):
        """
        Forward pass of Layer Normalization for one layer.

        Z: (batch, features)
        Returns:
          Z_norm — normalized tensor
          mean, var, std — statistics over features (axis=1) for backprop
        """
        mean = Z.mean(axis=1, keepdims=True)
        var = ((Z - mean) ** 2).mean(axis=1, keepdims=True)
        std = xp.sqrt(var + eps)
        Z_norm = (Z - mean) / std
        return Z_norm, mean, var, std

    @staticmethod
    def layer_norm_backward(dZ_norm, Z, mean, var, std, eps=1e-5):
        """
        Backward pass for Layer Normalization.

        dZ_norm — gradient w.r.t. the normalized output Z_norm.
        Output: dZ — gradient w.r.t. the input Z.
        """
        # Number of features for normalization
        N = Z.shape[1]
        Z_centered = Z - mean

        # dL/dvar
        dvar = xp.sum(
            dZ_norm * Z_centered * -0.5 * (var + eps) ** (-3.0 / 2.0),
            axis=1,
            keepdims=True,
        )
        # dL/dmean
        dmean = xp.sum(-dZ_norm / std, axis=1, keepdims=True) + dvar * xp.mean(
            -2.0 * Z_centered, axis=1, keepdims=True
        )

        # dL/dZ
        dZ = dZ_norm / std + dvar * 2.0 * Z_centered / N + dmean / N
        return dZ

    # -------------------------- Losses --------------------------

    @staticmethod
    def bce_loss(y, yhat):
        """
        Binary cross-entropy:
        L = - mean( y * log(yhat) + (1 - y) * log(1 - yhat) )
        """
        eps = 1e-9
        return float(xp.mean(
            -y * xp.log(yhat + eps) - (1 - y) * xp.log(1 - yhat + eps)
        ))

    @staticmethod
    def bce_grad(y, yhat):
        """
        Gradient of bce_loss w.r.t. the output yhat.

        For a sigmoid output and BCE loss combined with backprop:
          dL/dZ = (yhat - y) / N
        Here we return dL/dA (A = yhat), so that after multiplying
        by the sigmoid derivative (A * (1 - A)) we obtain (yhat - y) / N.
        """
        eps = 1e-9
        N = y.shape[0]
        # dL/dA so that after multiplying by the sigmoid derivative we get (yhat - y) / N
        return (yhat - y) / ((yhat * (1 - yhat)) + eps) / max(N, 1)

    @staticmethod
    def ce_loss(y, yhat):
        """
        Cross-entropy for the multiclass case (one-hot y):
        L = - mean( sum_c y_c * log(yhat_c) )
        """
        eps = 1e-9
        return float(-xp.mean(xp.sum(y * xp.log(yhat + eps), axis=1)))

    @staticmethod
    def ce_grad(y, yhat):
        """
        Gradient of CE + softmax:
        dL/dZ = (yhat - y) / N, so here we return exactly that form.
        """
        return (yhat - y) / y.shape[0]

    @staticmethod
    def mse_loss(y, yhat):
        # Mean squared error: mean( (yhat - y)^2 )
        return float(xp.mean((yhat - y) ** 2))

    # ------------------------ Constructor ------------------------

    def __init__(self,
                 layers,
                 init="xavier_normal",
                 use_layernorm: bool = True,
                 ln_every_k: int = 1):
        """
        layers: list of layer sizes, e.g. [105, 512, 256, 1]
        use_layernorm: whether to use LayerNorm on hidden layers
        ln_every_k: apply LayerNorm on every k-th layer (1 = every layer,
                    2 = every second, <=0 = disable LN)
        """
        self.layers = layers
        self.init_method = init
        self.params = {}  # W_i, b_i
        self.cache = {}   # intermediate values for backprop
        self.m = {}       # first moment for Adam
        self.v = {}       # second moment for Adam
        self.t = 0        # Adam step

        self.use_layernorm = bool(use_layernorm)
        self.ln_every_k = int(ln_every_k) if ln_every_k is not None else 0

        # Initialize weights and moments
        for i in range(1, len(layers)):
            fan_in = layers[i - 1]
            fan_out = layers[i]

            self.params[f"W{i}"] = Network.init_weights(fan_in, fan_out, init)
            self.params[f"b{i}"] = xp.zeros((1, fan_out), dtype=xp.float32)

            self.m[f"W{i}"] = xp.zeros_like(self.params[f"W{i}"])
            self.v[f"W{i}"] = xp.zeros_like(self.params[f"W{i}"])
            self.m[f"b{i}"] = xp.zeros_like(self.params[f"b{i}"])
            self.v[f"b{i}"] = xp.zeros_like(self.params[f"b{i}"])

    # ------------------------ helper ------------------------

    def _use_ln_for_layer(self, i: int) -> bool:
        """
        Checks whether LayerNorm should be applied for layer number i.
        LN is applied only to hidden layers according to ln_every_k.
        """
        if not self.use_layernorm:
            return False
        if self.ln_every_k <= 0:
            return False
        return (i % self.ln_every_k) == 0

    # ------------------------ Forward pass ------------------------

    def forward(self, X, hidden_act="relu", output_act="sigmoid"):
        """
        Forward pass of the network.

        hidden_act  — activation on hidden layers
        output_act  — activation of the output layer
        """
        A = X
        self.cache = {"A0": A}
        L = len(self.layers) - 1  # number of layers with weights

        for i in range(1, L + 1):
            # Linear part: Z_raw = A_{i-1} * W_i + b_i
            Z_raw = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            self.cache[f"Z_raw{i}"] = Z_raw

            if i == L:  # output layer without LayerNorm
                if output_act == "sigmoid":
                    A = Network.sigmoid(Z_raw)
                elif output_act == "softmax":
                    A = Network.softmax(Z_raw)
                elif output_act == "tanh":
                    A = Network.tanh(Z_raw)
                elif output_act == "relu":
                    A = Network.relu(Z_raw)
                elif output_act == "gelu":
                    A = Network.gelu(Z_raw)
                else:
                    # Linear output (no activation)
                    A = Z_raw
                self.cache[f"A{i}"] = A
                return A

            # hidden layers: with LayerNorm or without
            if self._use_ln_for_layer(i):
                # LN over features within the batch
                Z_norm, mean, var, std = Network.layer_norm_forward(Z_raw)
                self.cache[f"LN_mean{i}"] = mean
                self.cache[f"LN_var{i}"] = var
                self.cache[f"LN_std{i}"] = std
            else:
                Z_norm = Z_raw
            self.cache[f"Z{i}"] = Z_norm

            # Nonlinearity on the hidden layer
            if hidden_act == "relu":
                A = Network.relu(Z_norm)
            elif hidden_act == "gelu":
                A = Network.gelu(Z_norm)
            elif hidden_act == "tanh":
                A = Network.tanh(Z_norm)
            elif hidden_act == "sigmoid":
                A = Network.sigmoid(Z_norm)
            else:
                # If activation is not set — keep it linear
                A = Z_norm

            self.cache[f"A{i}"] = A

        return A

    # ------------------------ Backward pass ------------------------

    def backward(self, y, yhat,
                 hidden_act="relu",
                 output_act="sigmoid",
                 loss_fn="bce"):
        """
        Backward pass (backpropagation).

        y      — ground-truth values
        yhat   — network output from forward()
        loss_fn — "bce", "ce" or "mse" (defines dL/dA for the output layer)
        """
        grads = {}
        L = len(self.layers) - 1

        # dA from the loss function (gradient w.r.t. the output of the last layer)
        if loss_fn == "bce":
            dA = Network.bce_grad(y, yhat)
        elif loss_fn == "ce":
            dA = Network.ce_grad(y, yhat)
        else:  # mse
            dA = (yhat - y) * 2.0 / y.shape[0]

        # Traverse layers in reverse order: L, L-1, ..., 1
        for i in reversed(range(1, L + 1)):
            A_prev = self.cache[f"A{i-1}"]
            Z_raw = self.cache[f"Z_raw{i}"]

            if i == L:
                # Output layer: depends on the chosen activation
                if output_act == "sigmoid":
                    dZ = dA * Network.sigmoid_deriv(yhat)
                elif output_act == "tanh":
                    dZ = dA * Network.tanh_deriv(yhat)
                elif output_act == "relu":
                    dZ = dA * Network.relu_deriv(Z_raw)
                elif output_act == "gelu":
                    dZ = dA * Network.gelu_deriv(Z_raw)
                elif output_act == "softmax":
                    # For softmax + CE: dZ = (yhat - y) / N is already accounted for
                    dZ = dA
                else:
                    # Linear output without activation
                    dZ = dA
            else:
                # Hidden layers
                Z_norm = self.cache[f"Z{i}"]

                # dA -> dZ_norm through the activation derivative
                if hidden_act == "relu":
                    dZ_norm = dA * Network.relu_deriv(Z_norm)
                elif hidden_act == "gelu":
                    dZ_norm = dA * Network.gelu_deriv(Z_norm)
                elif hidden_act == "tanh":
                    dZ_norm = dA * Network.tanh_deriv(self.cache[f"A{i}"])
                elif hidden_act == "sigmoid":
                    dZ_norm = dA * Network.sigmoid_deriv(self.cache[f"A{i}"])
                else:
                    dZ_norm = dA

                # if LayerNorm was used on this layer — do LN backward,
                # otherwise just dZ = dZ_norm
                if self._use_ln_for_layer(i):
                    mean = self.cache[f"LN_mean{i}"]
                    var = self.cache[f"LN_var{i}"]
                    std = self.cache[f"LN_std{i}"]
                    dZ = Network.layer_norm_backward(dZ_norm, Z_raw, mean, var, std)
                else:
                    dZ = dZ_norm

            # Gradients w.r.t. W_i and b_i
            grads[f"W{i}"] = A_prev.T @ dZ
            grads[f"b{i}"] = xp.sum(dZ, axis=0, keepdims=True)

            # dA for the previous layer (i-1)
            dA = dZ @ self.params[f"W{i}"].T

        return grads

    # ------------------------ Adam update ------------------------

    def adam_update(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Update parameters using the Adam algorithm.

        grads — gradient dictionary for W_i, b_i
        lr    — learning rate
        """
        self.t += 1  # optimizer step

        for key in grads:
            # Update first and second moments
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            # Parameter update
            self.params[key] -= lr * m_hat / (xp.sqrt(v_hat) + eps)

    # ------------------------ FIT (batches + metrics + logging) ------------------------

    def fit(self,
            Xtr, Ytr,
            Xva, Yva,
            hidden_activation="relu",
            output_activation="sigmoid",
            loss="bce",
            optimizer="adam",
            lr=1e-3,
            batch_size=128,
            max_epochs=30,
            early_stopping=True,
            patience=5,
            min_delta=1e-4,
            debug_show: bool = True,
            save_model: bool = False,
            model_type: str | None = None,
            config_dict: dict | None = None,
            log_dir: str | None = "logs",
            plot_metrics: bool = False,
            plots_dir: str | None = None):
        """
        Main training loop.

        Performs:
          - mini-batch training (shuffling indices);
          - loss and metric computation on train/val;
          - early stopping based on val_loss;
          - optional saving of the model and logs;
          - optional plotting of metric curves.
        """

        # move to GPU if needed
        if GPU_AVAILABLE:
            Xtr = xp.asarray(Xtr, dtype=xp.float32)
            Ytr = xp.asarray(Ytr, dtype=xp.float32)
            Xva = xp.asarray(Xva, dtype=xp.float32)
            Yva = xp.asarray(Yva, dtype=xp.float32)

        n_train = Xtr.shape[0]

        # detector: (N,) or (N,1); classifier: (N, C) with C>1
        is_binary = (Ytr.ndim == 1) or (Ytr.ndim == 2 and Ytr.shape[1] == 1)

        # Metric history for plotting / analysis
        if is_binary:
            history = {
                "epoch": [],
                "train_loss": [],
                "val_loss": [],
                "val_acc": [],
                "val_precision": [],
                "val_recall": [],
                "val_f1": [],
            }
        else:
            history = {
                "epoch": [],
                "train_loss": [],
                "val_loss": [],
                "val_acc": [],
                "val_macro_f1": [],
            }

        # in-memory log (to write to .txt at the end)
        log_lines = []

        # base filename: <type>_h<number of hidden layers>
        hidden_layers = max(0, len(self.layers) - 2)
        mt = model_type if model_type is not None else "model"
        base_name = f"{mt}_h{hidden_layers}"

        # ------------------------ EPOCH 0: metrics BEFORE training ------------------------

        # train loss @ epoch 0
        out_tr0 = self.forward(Xtr, hidden_activation, output_activation)
        if loss == "bce":
            train_loss0 = Network.bce_loss(Ytr, out_tr0)
        elif loss == "ce":
            train_loss0 = Network.ce_loss(Ytr, out_tr0)
        else:
            train_loss0 = Network.mse_loss(Ytr, out_tr0)

        # val loss @ epoch 0
        out_va0 = self.forward(Xva, hidden_activation, output_activation)
        if loss == "bce":
            val_loss0 = Network.bce_loss(Yva, out_va0)
        elif loss == "ce":
            val_loss0 = Network.ce_loss(Yva, out_va0)
        else:
            val_loss0 = Network.mse_loss(Yva, out_va0)

        # validation metrics @ epoch 0
        if GPU_AVAILABLE:
            y_true_val0 = np.asarray(Yva.get())
            y_pred_val0 = np.asarray(out_va0.get())
        else:
            y_true_val0 = np.asarray(Yva)
            y_pred_val0 = np.asarray(out_va0)

        history["epoch"].append(0)
        history["train_loss"].append(float(train_loss0))
        history["val_loss"].append(float(val_loss0))

        if is_binary:
            # For the binary case we use a 0.5 threshold
            y_true_flat0 = y_true_val0.reshape(-1).astype(int)
            y_pred_flat0 = y_pred_val0.reshape(-1)
            preds0 = (y_pred_flat0 >= 0.5).astype(int)

            acc0 = float(np.mean(preds0 == y_true_flat0))
            tp0 = np.sum((preds0 == 1) & (y_true_flat0 == 1))
            fp0 = np.sum((preds0 == 1) & (y_true_flat0 == 0))
            fn0 = np.sum((preds0 == 0) & (y_true_flat0 == 1))

            prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
            rec0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
            if prec0 + rec0 > 0:
                f10 = 2 * prec0 * rec0 / (prec0 + rec0)
            else:
                f10 = 0.0

            history["val_acc"].append(acc0)
            history["val_precision"].append(prec0)
            history["val_recall"].append(rec0)
            history["val_f1"].append(f10)

            msg0 = (
                f"[Epoch 000/{max_epochs}] "
                f"train_loss={train_loss0:.6f}  "
                f"val_loss={val_loss0:.6f}  "
                f"val_acc={acc0:.4f}  "
                f"val_prec={prec0:.4f}  "
                f"val_rec={rec0:.4f}  "
                f"val_f1={f10:.4f}"
            )
        else:
            # multiclass metrics (accuracy + macro-F1)
            acc0 = accuracy_mc(y_pred_val0, y_true_val0)
            macro_f10 = macro_f1_mc(y_pred_val0, y_true_val0)

            history["val_acc"].append(acc0)
            history["val_macro_f1"].append(macro_f10)

            msg0 = (
                f"[Epoch 000/{max_epochs}] "
                f"train_loss={train_loss0:.6f}  "
                f"val_loss={val_loss0:.6f}  "
                f"val_acc={acc0:.4f}  "
                f"val_macroF1={macro_f10:.4f}  "
            )

        if debug_show:
            print(msg0)
        log_lines.append(msg0 + "\n")

        # ------------- initialize early stopping baseline (epoch 0) -------------
        best_val = float(val_loss0)  # best val_loss so far
        no_imp = 0                   # number of epochs without improvement
        # Store a copy of the best parameters on CPU
        best_params = {
            k: (v.get() if GPU_AVAILABLE else np.array(v, copy=True))
            for k, v in self.params.items()
        }

        # ------------------------ TRAINING (epochs 1..max_epochs) ------------------------

        for epoch in range(1, max_epochs + 1):
            # --------- TRAIN: standard shuffled mini-batches ---------
            idx = xp.random.permutation(n_train)
            train_loss_sum = 0.0
            seen = 0  # number of samples processed by batched training

            for start in range(0, n_train, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = Xtr[batch_idx]
                Yb = Ytr[batch_idx]

                # Forward pass on the mini-batch
                out = self.forward(Xb, hidden_activation, output_activation)

                # Compute batch loss
                if loss == "bce":
                    L_batch = Network.bce_loss(Yb, out)
                elif loss == "ce":
                    L_batch = Network.ce_loss(Yb, out)
                else:
                    L_batch = Network.mse_loss(Yb, out)

                bs = Xb.shape[0]
                train_loss_sum += L_batch * bs
                seen += bs

                # Backward pass and weight update
                grads = self.backward(Yb, out,
                                      hidden_activation,
                                      output_activation,
                                      loss)

                if optimizer == "adam":
                    self.adam_update(grads, lr)
                else:
                    # Simple SGD if Adam is not selected
                    for k in self.params:
                        self.params[k] -= lr * grads[k]

            # Average train_loss for the epoch
            train_loss = train_loss_sum / max(seen, 1)

            # --------- VALIDATION + METRICS ---------
            out_va = self.forward(Xva, hidden_activation, output_activation)

            if loss == "ce":
                val_loss = Network.ce_loss(Yva, out_va)
            elif loss == "bce":
                val_loss = Network.bce_loss(Yva, out_va)
            else:
                val_loss = Network.mse_loss(Yva, out_va)

            # move to CPU for metrics.py
            if GPU_AVAILABLE:
                y_true_val = np.asarray(Yva.get())
                y_pred_val = np.asarray(out_va.get())
            else:
                y_true_val = np.asarray(Yva)
                y_pred_val = np.asarray(out_va)

            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if is_binary:
                # Binary metrics
                y_true_flat = y_true_val.reshape(-1).astype(int)
                y_pred_flat = y_pred_val.reshape(-1)

                preds = (y_pred_flat >= 0.5).astype(int)

                acc_val = float(np.mean(preds == y_true_flat))

                tp = np.sum((preds == 1) & (y_true_flat == 1))
                fp = np.sum((preds == 1) & (y_true_flat == 0))
                fn = np.sum((preds == 0) & (y_true_flat == 1))

                precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if precision_val + recall_val > 0:
                    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
                else:
                    f1_val = 0.0

                history["val_acc"].append(acc_val)
                history["val_precision"].append(precision_val)
                history["val_recall"].append(recall_val)
                history["val_f1"].append(f1_val)

                msg = (
                    f"[Epoch {epoch:03d}/{max_epochs}] "
                    f"train_loss={train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"val_acc={acc_val:.4f}  "
                    f"val_prec={precision_val:.4f}  "
                    f"val_rec={recall_val:.4f}  "
                    f"val_f1={f1_val:.4f}"
                )
            else:
                # multiclass metrics
                val_acc = accuracy_mc(y_pred_val, y_true_val)
                val_macro_f1 = macro_f1_mc(y_pred_val, y_true_val)

                history["val_acc"].append(val_acc)
                history["val_macro_f1"].append(val_macro_f1)

                msg = (
                    f"[Epoch {epoch:03d}/{max_epochs}] "
                    f"train_loss={train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"val_acc={val_acc:.4f}  "
                    f"val_macroF1={val_macro_f1:.4f}  "
                )

            if debug_show:
                print(msg)
            log_lines.append(msg + "\n")

            # --------- Early stopping by val_loss with checkpointing ---------
            # 1) Always update best_params if val_loss improved
            if val_loss + min_delta < best_val:
                best_val = float(val_loss)
                no_imp = 0
                # store weights on CPU as numpy
                best_params = {
                    k: (v.get() if GPU_AVAILABLE else np.array(v, copy=True))
                    for k, v in self.params.items()
                }
            else:
                # 2) Use the patience counter only if early stopping is enabled
                if early_stopping:
                    no_imp += 1
                    if no_imp >= patience:
                        es_msg = f"[Early stopping @ epoch {epoch}]"
                        if debug_show:
                            print(es_msg)
                        log_lines.append(es_msg + "\n")
                        # restore the best weights
                        if best_params is not None:
                            for k in self.params:
                                self.params[k] = xp.asarray(
                                    best_params[k], dtype=xp.float32
                                )
                        break

        # if early stopping did not trigger, but best_params exists — still restore the best ones
        if best_params is not None:
            for k in self.params:
                self.params[k] = xp.asarray(best_params[k], dtype=xp.float32)

        # ------------------------ SAVE MODEL AND LOGS ------------------------

        if save_model:
            # prepare directories
            if log_dir is None:
                log_dir = "."
            os.makedirs(log_dir, exist_ok=True)

            # model path
            model_path = os.path.join(log_dir, base_name + ".npz")
            self.save(model_path)

            # text log path
            log_path = os.path.join(log_dir, base_name + ".txt")

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=== MODEL INFO ===\n")
                f.write(f"model_type = {mt}\n")
                f.write(f"layers = {self.layers}\n")
                f.write(f"hidden_layers = {hidden_layers}\n")
                f.write(f"use_layernorm = {self.use_layernorm}\n")
                f.write(f"ln_every_k = {self.ln_every_k}\n")
                f.write("\n")

                if config_dict is not None:
                    f.write("=== CONFIG FROM main.py ===\n")
                    for k in sorted(config_dict.keys()):
                        f.write(f"{k} = {config_dict[k]}\n")
                    f.write("\n")

                f.write("=== TRAINING LOG ===\n")
                for line in log_lines:
                    f.write(line)

        # ------------------------ PLOT AND SAVE METRIC CURVES ------------------------

        if plot_metrics:
            import matplotlib.pyplot as plt  # local import

            # directory for plots
            if plots_dir is None:
                base_plots_dir = log_dir if log_dir is not None else "."
                plots_dir = os.path.join(base_plots_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            epochs = history["epoch"]

            # 1) Loss (train vs val)
            plt.figure()
            plt.plot(epochs, history["train_loss"], label="train_loss")
            plt.plot(epochs, history["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss ({base_name})")
            plt.legend()
            loss_path = os.path.join(plots_dir, base_name + "_loss.png")
            plt.savefig(loss_path)
            plt.close()

            # 2) Other metrics
            if is_binary:
                plt.figure()
                plt.plot(epochs, history["val_acc"], label="val_acc")
                plt.plot(epochs, history["val_f1"], label="val_f1")
                plt.xlabel("Epoch")
                plt.ylabel("Metric")
                plt.title(f"Validation metrics ({base_name})")
                plt.legend()
                m_path = os.path.join(plots_dir, base_name + "_metrics.png")
                plt.savefig(m_path)
                plt.close()
            else:
                plt.figure()
                plt.plot(epochs, history["val_acc"], label="val_acc")
                plt.plot(epochs, history["val_macro_f1"], label="val_macroF1")
                plt.xlabel("Epoch")
                plt.ylabel("Metric")
                plt.title(f"Validation metrics ({base_name})")
                plt.legend()
                m_path = os.path.join(plots_dir, base_name + "_metrics.png")
                plt.savefig(m_path)
                plt.close()

        return history

    # ------------------------ Predict ------------------------

    def predict(self, X, hidden_activation="relu", output_activation="sigmoid"):
        """
        Inference (forward pass) without training.
        Returns a numpy array even when using the GPU.
        """
        if GPU_AVAILABLE:
            X = xp.asarray(X, dtype=xp.float32)
        out = self.forward(X, hidden_activation, output_activation)
        return out.get() if GPU_AVAILABLE else out

    # ------------------------ SAVE / LOAD ------------------------

    def save(self, filepath: str):
        """
        Save the model to a .npz file:
          - layers
          - use_layernorm, ln_every_k
          - all W_i, b_i

        filepath — full path including extension, e.g. "detector_40layers.npz"
        """
        # weights on CPU
        params_cpu = {
            k: (v.get() if GPU_AVAILABLE else np.asarray(v))
            for k, v in self.params.items()
        }

        # Model metadata
        meta = {
            "layers": np.asarray(self.layers, dtype=np.int32),
            "use_layernorm": np.asarray(
                [1 if self.use_layernorm else 0], dtype=np.int8
            ),
            "ln_every_k": np.asarray([self.ln_every_k], dtype=np.int32),
        }

        np.savez(filepath, **meta, **params_cpu)

    @classmethod
    def from_file(cls, filepath: str):
        """
        Create a Network instance from a file saved with save().

        NOTE: before calling, it is recommended to run set_backend(USE_GPU) so that
        weights are placed on the desired backend (CPU/GPU) immediately.
        """
        with np.load(filepath, allow_pickle=False) as data:
            if "layers" not in data:
                raise ValueError("Saved model file does not contain 'layers' array.")

            # Restore the architecture
            layers = data["layers"].astype(int).tolist()

            use_ln = True
            ln_k = 1
            if "use_layernorm" in data:
                use_ln = bool(int(data["use_layernorm"][0]))
            if "ln_every_k" in data:
                ln_k = int(data["ln_every_k"][0])

            # Create the network with the same LN settings
            net = cls(layers, init="he", use_layernorm=use_ln, ln_every_k=ln_k)

            # load weights
            for key in net.params.keys():
                if key not in data.files:
                    raise ValueError(f"Parameter '{key}' not found in file.")
                arr = data[key]
                net.params[key] = xp.asarray(arr, dtype=xp.float32)
                net.m[key] = xp.zeros_like(net.params[key])
                net.v[key] = xp.zeros_like(net.params[key])

        return net
