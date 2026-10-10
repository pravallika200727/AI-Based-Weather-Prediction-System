
import csv
import json
from urllib.parse import urlencode
from urllib.request import urlopen

print("AI WEATHER PREDICTION SYSTEM")
print("Downloading real historical weather data...")

params = {
    "latitude": 17.385,
    "longitude": 78.487,
    "start_date": "2016-01-01",
    "end_date": "2025-12-31",
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "pressure_msl,"
        "wind_speed_10m,"
        "precipitation"
    ),
    "timezone": "Asia/Kolkata"
}

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    + urlencode(params)
)

with urlopen(url, timeout=180) as response:
    data = json.loads(response.read().decode("utf-8"))

hourly = data["hourly"]

with open("weather.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "date", "temperature", "humidity",
        "pressure", "wind_speed", "precipitation"
    ])

    count = 0

    for i, timestamp in enumerate(hourly["time"]):
        values = [
            hourly["temperature_2m"][i],
            hourly["relative_humidity_2m"][i],
            hourly["pressure_msl"][i],
            hourly["wind_speed_10m"][i],
            hourly["precipitation"][i]
        ]

        if all(v is not None for v in values):
            writer.writerow([timestamp] + values)
            count += 1

print("\nDownload completed!")
print("Location: Hyderabad, India")
print("Period: 2016 to 2025")
print("Valid hourly records:", count)
print("Saved to weather.csv")
