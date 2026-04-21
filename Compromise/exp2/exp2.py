# Implementation of Linear Regression.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1 & 2: Import libraries and dataset
df = pd.read_csv(
    r"C:\Users\Sumit Patel\PycharmProjects\ML_Experiments\datasets\weather.csv"
)

# Step 3: Assign independent (X) and dependent (Y) variables
X = df["MaxTemp"].values
Y = df["Temp3pm"].values

# Step 4: Calculate slope and y-intercept using Least Squares Method
mean_X, mean_Y = np.mean(X), np.mean(Y)
slope = np.sum((X - mean_X) * (Y - mean_Y)) / np.sum((X - mean_X) ** 2)
y_intercept = mean_Y - slope * mean_X
Y_pred = slope * X + y_intercept

print(f"Slope (m): {slope:.4f}")
print(f"Y-intercept (c): {y_intercept:.4f}")

# Step 5: Plotting the line of best fit
plt.scatter(X, Y, color="blue", alpha=0.5, label="Actual Data")
plt.plot(
    X, Y_pred, color="red", linewidth=2, label=f"Y = {slope:.2f}X + {y_intercept:.2f}"
)
plt.xlabel("Max Temperature (°C)")
plt.ylabel("3PM Temperature (°C)")
plt.title("Linear Regression - MaxTemp vs Temp3pm")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(
    r"C:\Users\Sumit Patel\PycharmProjects\ML_Experiments\exp8\linear_regression_plot.png",
    dpi=150,
)

# Step 6: Model Evaluation (RMSE and R-squared)
rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
r_squared = 1 - np.sum((Y - Y_pred) ** 2) / np.sum((Y - mean_Y) ** 2)
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r_squared:.4f}")
