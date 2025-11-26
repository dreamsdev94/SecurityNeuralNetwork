from data_utils import create_dataset
from network import Network
from metrics import accuracy


def main():

    Xtr, Xva, Ytr, Yva, meta = create_dataset(
        "Train_Test_Windows_10.csv",
        scaling="zscore",
        return_meta=True
    )

    layers = [Xtr.shape[1], 64, 32, 8]
    net = Network(layers)

    hist = net.fit(
        Xtr, Ytr, Xva, Yva,
        loss="auto",
        max_epochs=80
    )

    yhat = net.predict(Xva)
    acc = accuracy(yhat, Yva)

    print(f"\nAccuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
