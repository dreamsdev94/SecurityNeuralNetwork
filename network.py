import numpy as np

# ==========================
# активації
# ==========================
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(np.float32)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_deriv(a): return a * (1 - a)

def tanh(x): return np.tanh(x)
def tanh_deriv(a): return 1 - a * a

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


# ==========================
# втрати
# ==========================
def bce_loss(y, yhat):
    eps = 1e-9
    return -np.mean(y * np.log(yhat + eps) + (1-y) * np.log(1-yhat + eps))

def bce_grad(y, yhat):
    eps = 1e-9
    return (yhat - y) / ((yhat*(1-yhat)) + eps)

def ce_loss(y, yhat):
    eps = 1e-9
    return -np.mean(np.sum(y*np.log(yhat + eps), axis=1))

def ce_grad(y, yhat):
    return (yhat - y) / y.shape[0]

def mse_loss(y, yhat):
    return np.mean((yhat - y)**2)

def mse_grad(y, yhat):
    return 2*(yhat - y) / y.size


# ==========================
# мережа
# ==========================
class Network:

    def __init__(self, layers):
        """
        layers = [input, h1, h2, ..., output]
        """
        self.layers = layers
        self.params = {}
        self.cache = {}

        for i in range(1, len(layers)):
            self.params[f"W{i}"] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2/layers[i-1])
            self.params[f"b{i}"] = np.zeros((1, layers[i]))

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, X, hidden_activation, output_activation):
        A = X
        self.cache["A0"] = X

        for i in range(1, len(self.layers)):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            self.cache[f"Z{i}"] = Z

            if i == len(self.layers)-1:   # output layer
                if output_activation == "sigmoid":
                    A = sigmoid(Z)
                elif output_activation == "softmax":
                    A = softmax(Z)
                elif output_activation == "tanh":
                    A = tanh(Z)
                else:
                    A = Z
            else:
                if hidden_activation == "relu":
                    A = relu(Z)
                elif hidden_activation == "tanh":
                    A = tanh(Z)
                elif hidden_activation == "sigmoid":
                    A = sigmoid(Z)

            self.cache[f"A{i}"] = A

        return A

    # ---------------------------------------------------------------------
    # backward
    # ---------------------------------------------------------------------
    def backward(self, Y, output, hidden_act, output_act, loss):
        grads = {}
        L = len(self.layers)-1

        if loss == "bce":
            dA = bce_grad(Y, output)
        elif loss == "ce":
            dA = ce_grad(Y, output)
        else:
            dA = mse_grad(Y, output)

        for i in reversed(range(1, L+1)):
            A_prev = self.cache[f"A{i-1}"]
            Z = self.cache[f"Z{i}"]

            if i == L:   # output layer
                if output_act == "sigmoid":
                    dZ = dA * sigmoid_deriv(output)
                elif output_act == "tanh":
                    dZ = dA * tanh_deriv(output)
                elif output_act == "softmax":
                    dZ = dA     # CE + Softmax = simple gradient
                else:
                    dZ = dA
            else:   # hidden
                A = self.cache[f"A{i}"]
                if hidden_act == "relu":
                    dZ = dA * relu_deriv(Z)
                elif hidden_act == "tanh":
                    dZ = dA * tanh_deriv(A)
                elif hidden_act == "sigmoid":
                    dZ = dA * sigmoid_deriv(A)

            grads[f"W{i}"] = A_prev.T @ dZ
            grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True)

            dA = dZ @ self.params[f"W{i}"].T

        return grads

    # ---------------------------------------------------------------------
    # update
    # ---------------------------------------------------------------------
    def update(self, grads, lr):
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] -= lr * grads[f"W{i}"]
            self.params[f"b{i}"] -= lr * grads[f"b{i}"]

    # ---------------------------------------------------------------------
    # train
    # ---------------------------------------------------------------------
    def fit(self,
            Xtr, Ytr,
            Xva, Yva,
            hidden_activation="relu",
            output_activation="sigmoid",
            loss="bce",
            batch_size=128,
            max_epochs=30,
            lr=0.001,
            verbose=True):

        hist = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "batch_step": [],
            "batch_loss": [],
            "batch_acc": [],
        }

        N = Xtr.shape[0]
        steps = int(np.ceil(N / batch_size))
        step_counter = 0

        for epoch in range(max_epochs):

            idx = np.random.permutation(N)
            Xtr = Xtr[idx]
            Ytr = Ytr[idx]

            for step in range(steps):
                Xb = Xtr[step*batch_size:(step+1)*batch_size]
                Yb = Ytr[step*batch_size:(step+1)*batch_size]

                out = self.forward(Xb, hidden_activation, output_activation)

                # loss
                if loss == "bce": L = bce_loss(Yb, out)
                elif loss == "ce": L = ce_loss(Yb, out)
                else: L = mse_loss(Yb, out)

                # acc
                if output_activation == "sigmoid":
                    pred = (out >= 0.5).astype(int)
                    acc = np.mean(pred == Yb)
                else:
                    pred = np.argmax(out, axis=1)
                    true = np.argmax(Yb, axis=1)
                    acc = np.mean(pred == true)

                hist["batch_step"].append(step_counter)
                hist["batch_loss"].append(L)
                hist["batch_acc"].append(acc)

                grads = self.backward(Yb, out, hidden_activation, output_activation, loss)
                self.update(grads, lr)

                step_counter += 1

            # end epoch
            out_tr = self.forward(Xtr, hidden_activation, output_activation)
            out_va = self.forward(Xva, hidden_activation, output_activation)

            if loss == "bce":
                L_tr = bce_loss(Ytr, out_tr)
                L_va = bce_loss(Yva, out_va)
            elif loss == "ce":
                L_tr = ce_loss(Ytr, out_tr)
                L_va = ce_loss(Yva, out_va)
            else:
                L_tr = mse_loss(Ytr, out_tr)
                L_va = mse_loss(Yva, out_va)

            # accuracy
            if output_activation == "sigmoid":
                acc_tr = np.mean((out_tr>=0.5).astype(int) == Ytr)
                acc_va = np.mean((out_va>=0.5).astype(int) == Yva)
            else:
                acc_tr = np.mean(np.argmax(out_tr,axis=1) == np.argmax(Ytr,axis=1))
                acc_va = np.mean(np.argmax(out_va,axis=1) == np.argmax(Yva,axis=1))

            hist["epoch"].append(epoch)
            hist["train_loss"].append(L_tr)
            hist["val_loss"].append(L_va)
            hist["train_acc"].append(acc_tr)
            hist["val_acc"].append(acc_va)

            if verbose:
                print(f"[Epoch {epoch}] loss={L_tr:.4f} val={L_va:.4f} acc={acc_tr:.3f} val_acc={acc_va:.3f}")

        return hist

    # ---------------------------------------------------------------------
    # predict
    # ---------------------------------------------------------------------
    def predict(self, X, hidden_activation="relu", output_activation="sigmoid"):
        return self.forward(X, hidden_activation, output_activation)
