# Logistic regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay

feature_cols = ["Area","Bedrooms","Bathrooms","Floors","YearBuilt"]
df = pd.read_csv("../../ML/Exp03/House_Price_Prediction_Dataset.csv", usecols=feature_cols + ["Price"])
df["PriceCategory"] = (df.Price > df.Price.median()).astype(int)

X = df[feature_cols]
y = df["PriceCategory"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", f"{roc_auc_score(y_test, y_proba):.4f}")
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.show()
