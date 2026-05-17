import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
from tensorflow.keras.models import load_model
import numpy as np
import openpyxl

# Load data
df = pd.read_csv("C:\\Users\\sanch\\Downloads\\hdp\\heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Better split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔥 Tuned Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.3),

    Dense(16, activation='relu'),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🔥 Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    min_lr=0.00001
)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=20,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
joblib.dump(model, "C:\\Users\\sanch\\Downloads\\hdp\\model.pkl")
joblib.dump(scaler, "C:\\Users\\sanch\\Downloads\\hdp\\scaler.pkl")

print("Model and scaler saved successfully!")
#joblib.dump(model, "C:\\Users\\sanch\\Downloads\\hdp\\heart_model.pkl")
#model.save("C:\\Users\\sanch\\Downloads\\hdp\\MY_MODEL.h5")
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cf = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cf).plot()
plt.title("MY MODEL")
plt.show()
# Metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# ACCURACY COMPARISON GRAPH
# -------------------------------

wb = openpyxl.load_workbook("C:\\Users\\sanch\\Downloads\\hdp\\output.xlsx")
sheet = wb.active

# Read a specific cell value into a variable
bestacc = sheet['A1'].value
plt.figure(figsize=(10, 5))
dict={"random forest":bestacc,"my model":acc}
plt.bar(dict.keys(),dict.values())

plt.title("Algorithm Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")

plt.xticks(rotation=15)

plt.show()


