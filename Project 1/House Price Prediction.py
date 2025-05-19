import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("carlmcbrideellis/house-prices-advanced-regression-solution-file")
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        csv_path = os.path.join(path, filename)
        break
df = pd.read_csv(csv_path)
df.head()
print(df)
print("Path to dataset files:", path)
df.info()

df.describe()

df.isnull().sum().sort_values(ascending=False).head(10)
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["SalePrice"], bins=30, kde=True)
plt.title("House Prices")

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

## Boston House Prices-Advanced Regression Techniques
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# X = df.drop("MEDV", axis=1)
# y = df["MEDV"]

X = np.array(df.drop("MEDV", axis=1), dtype=float)
y = np.array(df["MEDV"], dtype=float)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf))
