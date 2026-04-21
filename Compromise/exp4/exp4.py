# SVM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def resolve_data_file(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / filename,
        script_dir.parent / filename,
        script_dir.parent.parent / "datasets" / filename,
        Path.cwd() / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find '{filename}'. Searched:\n{searched}")


def plot_boundary(model, x_sc, y_true, title, acc):
    x1, x2 = np.meshgrid(
        np.arange(x_sc[:, 0].min() - 0.5, x_sc[:, 0].max() + 0.5, 0.01),
        np.arange(x_sc[:, 1].min() - 0.5, x_sc[:, 1].max() + 0.5, 0.01),
    )
    z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
    labels, colors, markers = ["Not Purchased", "Purchased"], ["#CC0000", "#0000CC"], ["o", "^"]

    plt.figure(figsize=(9, 7))
    plt.contourf(x1, x2, z, alpha=0.3, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]))
    for c in (0, 1):
        plt.scatter(x_sc[y_true == c, 0], x_sc[y_true == c, 1], c=colors[c], edgecolors="white", s=50, marker=markers[c], label=labels[c])
    plt.title(f"SVM (RBF) - {title} | Accuracy: {acc * 100:.2f}%")
    plt.xlabel("Age (scaled)")
    plt.ylabel("Estimated Salary (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"svm_{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()


def main():
    csv_path = resolve_data_file("Social_Network_Ads.csv")
    data = pd.read_csv(csv_path)
    x = data[["Age", "EstimatedSalary"]].values
    y = data["Purchased"].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    scaler = StandardScaler()
    x_train_sc, x_test_sc = scaler.fit_transform(x_train), scaler.transform(x_test)

    model = SVC(kernel="rbf", random_state=0).fit(x_train_sc, y_train)
    y_pred = model.predict(x_test_sc)
    cm, acc = confusion_matrix(y_test, y_pred), model.score(x_test_sc, y_test)

    print(f"Shape: {data.shape} | Support Vectors: {sum(model.n_support_)} | Accuracy: {acc * 100:.2f}%")
    print(f"Class Distribution:\n{data['Purchased'].value_counts()}\nConfusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=["Not Purchased", "Purchased"]))

    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Not Purchased", "Purchased"],
        cmap="Blues",
        colorbar=True,
    )
    plt.title("Confusion Matrix - SVM")
    plt.tight_layout()
    plt.savefig("svm_confusion_matrix.png", dpi=150)
    plt.show()

    plot_boundary(model, x_train_sc, y_train, "Training Set", acc)
    plot_boundary(model, x_test_sc, y_test, "Test Set", acc)
    print("\nEXPERIMENT 4 - COMPLETE")


if __name__ == "__main__":
    main()
