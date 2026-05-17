import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("C:\\Users\\sanch\\Downloads\\hdp\\heart.csv")

print("\nFirst 5 Rows:\n")
print(df.head())

# -------------------------------
# CORRELATION MATRIX
# -------------------------------

plt.figure(figsize=(12, 10))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()

# -------------------------------
# FEATURES AND TARGET
# -------------------------------

X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# FEATURE SCALING
# -------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# MODELS
# -------------------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB()
}

results = {}

best_accuracy = 0
best_model = None
best_model_name = ""

# -------------------------------
# TRAIN MODELS
# -------------------------------

for name, model in models.items():

    print("\n" + "="*50)
    print(f"MODEL: {name}")
    print("="*50)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# -------------------------------
# SAVE BEST MODEL
# -------------------------------

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# -------------------------------
# ACCURACY COMPARISON GRAPH
# -------------------------------

plt.figure(figsize=(10, 5))

plt.bar(results.keys(), results.values())

plt.title("Algorithm Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")

plt.xticks(rotation=15)

plt.show()

# -------------------------------
# BEST MODEL RESULT
# -------------------------------

print("\n" + "="*50)
print("BEST MODEL")
print("="*50)

print(f"Model Name: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")



wb = openpyxl.Workbook() 
sheet = wb.active

# 2. Store the variable in a specific cell
sheet['A1'] = best_accuracy

# 3. Save the file
wb.save("C:\\Users\\sanch\\Downloads\\hdp\\output.xlsx")