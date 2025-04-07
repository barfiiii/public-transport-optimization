import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Define dummy values
routes = [101, 102, 103, 104, 105]
stops = ["Stop A", "Stop B", "Stop C", "Stop D"]
days = pd.date_range(start="2023-03-01", periods=5, freq='D')
time_slots = [(7, 9), (12, 14), (17, 19)]  # Morning, Noon, Evening

data = []

# Generate data
for day in days:
    for route in routes:
        for stop in stops:
            for slot in time_slots:
                hour = random.randint(slot[0], slot[1] - 1)
                minute = random.randint(0, 59)
                arrival = datetime(day.year, day.month, day.day, hour, minute)
                departure = arrival + timedelta(minutes=random.randint(3, 10))
                passengers = random.randint(5, 50)

                data.append({
                    "Route_ID": route,
                    "Stop_Name": stop,
                    "Arrival_Time": arrival.strftime("%Y-%m-%d %H:%M"),
                    "Departure_Time": departure.strftime("%Y-%m-%d %H:%M"),
                    "Passenger_Count": passengers
                })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("data/city_transport.csv", index=False)

print("✅ Dummy data generated and saved to data/city_transport.csv")
