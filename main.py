import numpy as np
import matplotlib.pyplot as plt

from data_utils import create_dataset
from network import Network


def plot_history(hist, title):
    plt.figure(figsize=(10,4))
    plt.plot(hist["epoch"], hist["train_loss"], label="train")
    plt.plot(hist["epoch"], hist["val_loss"], label="val")
    plt.title(f"{title} loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(hist["epoch"], hist["train_acc"], label="train")
    plt.plot(hist["epoch"], hist["val_acc"], label="val")
    plt.title(f"{title} accuracy")
    plt.legend()
    plt.grid()
    plt.show()


def main():

    det, cls = create_dataset(
        "Train_Test_Windows_10.csv",
        scaling_detector="minmax",
        scaling_classifier="zscore"
    )

    # ==========================
    # detector
    # ==========================
    det_model = Network([det.X_train.shape[1], 64, 32, 1])
    hist_det = det_model.fit(
        det.X_train, det.y_train,
        det.X_val, det.y_val,
        hidden_activation="sigmoid",
        output_activation="sigmoid",
        loss="bce",
        lr=1e-3,
        max_epochs=20
    )
    plot_history(hist_det, "Detector")

    # ==========================
    # classifier
    # ==========================
    cls_model = Network([cls.X_train.shape[1], 128, 64, cls.y_train.shape[1]])
    hist_cls = cls_model.fit(
        cls.X_train, cls.y_train,
        cls.X_val, cls.y_val,
        hidden_activation="tanh",
        output_activation="softmax",
        loss="ce",
        lr=1e-3,
        max_epochs=25
    )
    plot_history(hist_cls, "Classifier")

    # ==========================
    # final accuracy
    # ==========================
    pred_det = (det_model.predict(det.X_val)>=0.5).astype(int)
    print("Detector accuracy:", np.mean(pred_det == det.y_val))

    pred_cls = np.argmax(cls_model.predict(cls.X_val), axis=1)
    print("Classifier accuracy:", np.mean(pred_cls == np.argmax(cls.y_val,axis=1)))


if __name__=="__main__":
    main()
