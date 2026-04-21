# Single Layer Perceptron Learning algorithm.

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=10):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0
        self.errors_per_epoch = []

    @staticmethod
    def _step(z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        _, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.n_epochs):
            errors = 0
            for x_i, target in zip(X, y):
                error = target - self._step(np.dot(x_i, self.weights) + self.bias)
                if error:
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    errors += abs(error)
            self.errors_per_epoch.append(errors)
        return self

    def predict(self, X):
        return self._step(np.dot(X, self.weights) + self.bias)


X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])

perceptron = Perceptron().fit(X_train, y_train)

print("Testing Perceptron for AND Gate:")
for x in X_train:
    print(f"Input: {x}, Prediction: {perceptron.predict(x)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x1 = np.linspace(-0.5, 1.5, 100)
x2 = -(perceptron.weights[0] * x1 + perceptron.bias) / perceptron.weights[1]

axes[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="red", marker="x", s=100, label="Class 0")
axes[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="blue", marker="o", s=100, label="Class 1")
axes[0].plot(x1, x2, "g-", label="Decision Boundary")
axes[0].set(xlabel="Input 1", ylabel="Input 2", title="Perceptron Decision Boundary (AND Gate)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(perceptron.errors_per_epoch) + 1), perceptron.errors_per_epoch, marker="o", color="purple")
axes[1].set(xlabel="Epoch", ylabel="Total Errors", title="Error Convergence Over Epochs")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
