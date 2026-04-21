# Principal Component Analysis..
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# If CSV is in the same folder: path = os.path.join(os.path.dirname(__file__), "House_Price_Prediction_Dataset.csv")
path = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "House_Price_Prediction_Dataset.csv"
)
df = pd.read_csv(path)

features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
X = df[features].values
y = df["Price"].values

X_std = StandardScaler().fit_transform(X)
n, d = X_std.shape

_, s, Vt = np.linalg.svd(X_std, full_matrices=False)
var_ratio = (s ** 2) / (n - 1)
var_ratio = var_ratio / var_ratio.sum()
cum_var = np.cumsum(var_ratio)

scores = X_std @ Vt.T
k = min(np.searchsorted(cum_var, 0.80) + 1, d)

sk_scores = PCA(n_components=d).fit_transform(X_std)
corr = abs(np.corrcoef(sk_scores[:, 0], scores[:, 0])[0, 1])

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
labels = [f"PC{i+1}" for i in range(d)]

ax[0, 0].bar(labels, var_ratio * 100, color="steelblue", edgecolor="navy")
ax[0, 0].set_title("Explained Variance per PC")
ax[0, 0].set_ylabel("Variance (%)")

ax[0, 1].plot(range(1, d + 1), cum_var * 100, "o-", color="darkorange", linewidth=2)
ax[0, 1].axhline(80, color="green", linestyle="--", label="80% threshold")
ax[0, 1].axvline(k, color="red", linestyle=":", label=f"k={k}")
ax[0, 1].set_title("Cumulative Explained Variance")
ax[0, 1].set_xlabel("Number of PCs")
ax[0, 1].set_ylabel("Cumulative (%)")
ax[0, 1].legend()

sc = ax[1, 0].scatter(scores[:, 0], scores[:, 1], c=y, cmap="plasma", alpha=0.6, s=20)
plt.colorbar(sc, ax=ax[1, 0], label="Price")
ax[1, 0].set_title(f"PC1 vs PC2 ({cum_var[1] * 100:.1f}% variance, corr={corr:.3f})")
ax[1, 0].set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
ax[1, 0].set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")

ax[1, 1].scatter(scores[:, 0], scores[:, 1], alpha=0.2, color="steelblue", s=10)
for i, f in enumerate(features):
    ax[1, 1].annotate("", xy=(Vt[0, i] * 3.5, Vt[1, i] * 3.5), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.5))
    ax[1, 1].text(Vt[0, i] * 4.0, Vt[1, i] * 4.0, f, ha="center", fontsize=9, fontweight="bold")
ax[1, 1].set_title("Biplot - PC1 & PC2 Loadings")
ax[1, 1].set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
ax[1, 1].set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")

plt.tight_layout()
plt.savefig("pca_svd_results.png", dpi=150, bbox_inches="tight")
plt.show()
