# train_emotion_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Generate synthetic features
def generate_synthetic_data(n=500):
    np.random.seed(42)
    sdnn = np.random.normal(loc=60, scale=20, size=n)
    gsr_mean = np.random.normal(loc=0.5, scale=0.2, size=n)

    labels = []
    for s, g in zip(sdnn, gsr_mean):
        if s > 80 and g < 0.4:
            labels.append("positive")
        elif s < 50 or g > 0.8:
            labels.append("stressed")
        else:
            labels.append("neutral")

    df = pd.DataFrame({
        "sdnn": sdnn,
        "gsr_mean": gsr_mean,
        "label": labels
    })
    return df

df = generate_synthetic_data()

# Train model
X = df[["sdnn", "gsr_mean"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "emotion_rf_model.pkl")
print("Model saved as emotion_rf_model.pkl")

# Evaluation
y_pred = model.predict(X)
print(classification_report(y, y_pred))
