# Hebbian Learning Algorithms.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HebbianLearning:
    def __init__(self, n_inputs, lr=0.01, epochs=50, rule="basic", decay=0.001):
        self.lr, self.epochs, self.rule, self.decay = lr, epochs, rule, decay
        self.w, self.b = np.zeros(n_inputs), 0.0
        self.weight_history, self.error_history = [], []

    def act(self, x): return np.where(x >= 0, 1, -1)
    def predict(self, X): return self.act(X @ self.w + self.b)
    def score(self, X, y): return np.mean(self.predict(X) == y)

    def _delta(self, x, y_hat):
        if self.rule == "basic": return x * y_hat, y_hat
        if self.rule == "decay": return x * y_hat - self.decay * self.w, y_hat - self.decay * self.b
        return y_hat * (x - y_hat * self.w), y_hat

    def fit(self, X, y):
        self.weight_history, self.error_history = [], []
        for _ in range(self.epochs):
            errors = 0
            for x, y_true in zip(X, y):
                y_hat = self.act(x @ self.w + self.b)
                dw, db = self._delta(x, y_hat)
                self.w += self.lr * dw
                self.b += self.lr * db
                errors += y_hat != y_true
            self.weight_history.append(self.w.copy())
            self.error_history.append(errors / len(X))


def load_data(path):
    df = pd.read_csv(path)
    y_col = df.columns[-1]
    x_cols = [c for c in df.columns[:-1] if df[c].dtype in ("object", "string")]
    df = pd.get_dummies(df, columns=x_cols, drop_first=True)
    X = StandardScaler().fit_transform(df.drop(columns=[y_col]).to_numpy())
    y = np.where(df[y_col].to_numpy() == 0, -1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot(models, names):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for m, n in zip(models, names): ax[0].plot(m.error_history, marker="o", ms=3, label=n)
    ax[0].set(title="Error Rate", xlabel="Epoch", ylabel="Error"); ax[0].grid(alpha=0.3); ax[0].legend()
    for m in models:
        for j in range(np.array(m.weight_history).shape[1]): ax[1].plot(np.array(m.weight_history)[:, j], alpha=0.6)
    ax[1].set(title="Weight Evolution", xlabel="Epoch", ylabel="Weight"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("hebbian_summary.png", dpi=150, bbox_inches="tight"); plt.show()


def main():
    # Dataset path
    # If CSV is in this same folder: Path(__file__).resolve().parent / "Social_Network_Ads.csv"
    data = Path(__file__).resolve().parents[2] / "ML" / "exp4" / "Social_Network_Ads.csv"
    X_train, X_test, y_train, y_test = load_data(data)
    names = ["Basic Hebbian", "Hebbian + Decay", "Oja's Rule"]
    models = [HebbianLearning(X_train.shape[1], rule=r) for r in ("basic", "decay", "oja")]
    for m, n in zip(models, names):
        m.fit(X_train, y_train)
        print(f"{n}: train={m.score(X_train, y_train):.4f}, test={m.score(X_test, y_test):.4f}")
    plot(models, names)


if __name__ == "__main__": main()
