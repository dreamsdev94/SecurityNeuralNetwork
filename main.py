import numpy as np
from data_utils import auto_normalize
from network import Network
from metrics import accuracy

###############################################################################
# Основний запуск навчання
###############################################################################

def train_model(layers, Xtr, Ytr, Xva, Yva,
                epochs=200, batch_size=128, lr=1e-3, loss="auto"):

    # тип задачі
    n_out = layers[-1]
    task_type = "binary" if n_out == 1 else "multiclass"

    # авто-режим
    if loss == "auto":

        if task_type == "binary":
            hidden_act = "sigmoid"
            output_act = "sigmoid"
            loss = "bce"

        else:
            hidden_act = "tanh"
            output_act = "softmax"
            loss = "ce"
    else:
        hidden_act = "tanh"
        output_act = "tanh"

    # глибокі мережі
    n_hidden = len(layers) - 2
    if n_hidden > 7:
        hidden_act = "relu"

    # ---- нормалізація ----
    Xtr_norm = auto_normalize(Xtr, task_type, hidden_act)
    Xva_norm = auto_normalize(Xva, task_type, hidden_act)

    # ---- запуск ----
    net = Network(layers)

    hist = net.fit(
        Xtr_norm, Ytr,
        Xva_norm, Yva,
        hidden_activation=hidden_act,
        output_activation=output_act,
        loss=loss,
        lr=lr,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=15,
        reduce_lr_on_plateau=0.5,
        verbose_every=epochs // 10
    )

    # ---- оцінка ----
    yhat_va = net.predict(
        Xva_norm,
        hidden_activation=hidden_act,
        output_activation=output_act
    )

    score = accuracy(yhat_va, Yva)

    return net, hist, score
