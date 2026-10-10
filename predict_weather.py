
import joblib
import pandas as pd

print("\n" + "=" * 55)
print("       AI-BASED WEATHER PREDICTION SYSTEM")
print("=" * 55)

location = "Hyderabad, Telangana"

try:
    # Load trained machine learning models
    temp_data = joblib.load("models/temperature_model.pkl")
    rain_data = joblib.load("models/rain_model.pkl")
    latest = joblib.load("models/latest_input.pkl")

    temp_model = temp_data["model"]
    rain_model = rain_data["model"]

    # Predict next-hour temperature and rainfall
    predicted_temp = temp_model.predict(latest)[0]
    rain_prediction = int(rain_model.predict(latest)[0])

    probabilities = rain_model.predict_proba(latest)[0]
    classes = list(rain_model.classes_)
    rain_probability = (
        probabilities[classes.index(1)] * 100
        if 1 in classes else 0
    )

    print(f"\nLocation: {location}")
    print("Forecast: Next Hour")
    print("-" * 55)

    print(f"Predicted Temperature : {predicted_temp:.1f} °C")
    print(
        "Rain Prediction      : "
        + ("Rain Expected" if rain_prediction == 1
           else "No Rain Expected")
    )
    print(f"Rain Probability      : {rain_probability:.1f}%")

    # Show latest observed weather values from the dataset
    data = pd.read_csv("weather.csv")
    data = data.dropna(subset=[
        "temperature", "humidity", "pressure",
        "wind_speed", "precipitation"
    ])
    latest_weather = data.iloc[-1]

    print("\nLatest Dataset Observations")
    print("-" * 55)
    print(f"Temperature : {latest_weather['temperature']:.1f} °C")
    print(f"Humidity    : {latest_weather['humidity']:.1f}%")
    print(f"Pressure    : {latest_weather['pressure']:.1f} hPa")
    print(f"Wind Speed  : {latest_weather['wind_speed']:.1f} km/h")
    print(f"Precipitation: {latest_weather['precipitation']:.2f} mm")

    print("\n" + "=" * 55)
    print("       Prediction completed successfully!")
    print("=" * 55)

except FileNotFoundError as error:
    print("\nRequired file not found:", error)
    print("Please download the dataset and train the models first.")
except Exception as error:
    print("\nError:", error)
    print("Please check your dataset and trained model files.")
