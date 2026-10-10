
import os
import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report
)

print("\nAI WEATHER PREDICTION - MODEL TRAINING")

# Load actual historical data
df = pd.read_csv("weather.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").drop_duplicates("date")
df = df.set_index("date")

columns = [
    "temperature", "humidity", "pressure",
    "wind_speed", "precipitation"
]

df[columns] = df[columns].apply(
    pd.to_numeric, errors="coerce"
)
df = df.dropna(subset=columns)

print("Dataset records:", len(df))

# Features available at the current hour
features = columns.copy()

# Previous-hour observations
for col in columns:
    for lag in [1, 3, 6, 24]:
        name = f"{col}_lag_{lag}"
        df[name] = df[col].shift(lag)
        features.append(name)

# Rolling temperature and humidity averages
for col in ["temperature", "humidity"]:
    name = f"{col}_mean_6h"
    df[name] = df[col].rolling(6).mean()
    features.append(name)

# Targets: the following hour
df["target_temperature"] = df["temperature"].shift(-1)
df["target_rain"] = (
    df["precipitation"].shift(-1) > 0.1
).astype(int)

df = df.dropna(subset=features + ["target_temperature"])
df = df.dropna(subset=["target_rain"])

if len(df) < 100:
    raise ValueError("Not enough usable records to train the model.")

# Keep chronological order to avoid future-data leakage
split = int(len(df) * 0.8)

X_train = df[features].iloc[:split]
X_test = df[features].iloc[split:]

y_temp_train = df["target_temperature"].iloc[:split]
y_temp_test = df["target_temperature"].iloc[split:]

y_rain_train = df["target_rain"].iloc[:split]
y_rain_test = df["target_rain"].iloc[split:]

# Train temperature prediction model
temp_model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

temp_model.fit(X_train, y_temp_train)
temp_pred = temp_model.predict(X_test)

mae = mean_absolute_error(y_temp_test, temp_pred)
rmse = mean_squared_error(
    y_temp_test, temp_pred
) ** 0.5

# Train rain prediction model
rain_model = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rain_model.fit(X_train, y_rain_train)
rain_pred = rain_model.predict(X_test)

accuracy = accuracy_score(y_rain_test, rain_pred)

# Save trained models and feature names
os.makedirs("models", exist_ok=True)

joblib.dump(
    {"model": temp_model, "features": features},
    "models/temperature_model.pkl"
)

joblib.dump(
    {"model": rain_model, "features": features},
    "models/rain_model.pkl"
)

# Save latest usable historical features for demo prediction
joblib.dump(
    X.iloc[[-1]] if False else df[features].iloc[[-1]],
    "models/latest_input.pkl"
)

print("\nTRAINING COMPLETED")
print("Algorithm: Random Forest")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTEMPERATURE EVALUATION")
print("MAE:", round(mae, 3), "degrees C")
print("RMSE:", round(rmse, 3), "degrees C")

print("\nRAIN PREDICTION EVALUATION")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nRain classification report:")
print(classification_report(
    y_rain_test,
    rain_pred,
    zero_division=0
))

print("Models saved in the models folder.")
