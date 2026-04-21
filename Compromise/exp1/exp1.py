# Data Preprocessing Techniques using NumPy, Pandas, and Matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("House_Price_Dataset.csv")

X = dataset.iloc[:, :-1].values  # Area
y = dataset.iloc[:, -1].values  # Price

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({"Actual Price": y_test, "Predicted Price": y_pred})
print(comparison)

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("House Price Prediction")
plt.xlabel("Area")
plt.ylabel("House Price")
plt.show()

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

sample_house = np.array([[2000]])
print("Predicted House Price:", regressor.predict(sample_house))
