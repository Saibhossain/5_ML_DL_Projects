import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("rhonarosecortez/telco-customer-churn")

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

df["ChurnScore"].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["ChurnScore"], bins=30, kde=True)
plt.title("Churn Score")

sns.countplot(data=df, x="ChurnScore")
now_df = df.drop(["CustomerID", "ChurnReason", "ChurnCategory", "CustomerStatus", "Country", "State", "City", "ZipCode"], axis=1, inplace=True)

print(new_df.shape)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["InternetType"].fillna("Unknown", inplace=True)
df["Offer"].fillna("None", inplace=True)
df.dropna(inplace=True)


# Encode target variable
df["ChurnLabel"] = df["ChurnLabel"].map({"Yes": 1, "No": 0})

from sklearn.preprocessing import LabelEncoder
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove("Under30")  # this is redundant with Age

# Encode binary columns with LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Drop "Under30" as it duplicates information
df.drop("Under30", axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X = df.drop("ChurnLabel", axis=1)
y = df["ChurnLabel"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

