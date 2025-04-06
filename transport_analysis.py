import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder if not exists
os.makedirs("plots", exist_ok=True)

# Load the CSV
df = pd.read_csv("data/city_transport.csv")

# Show basic info
print("📄 Data Preview:")
print(df.head())

# Convert time columns to datetime
df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
df['Departure_Time'] = pd.to_datetime(df['Departure_Time'])

# Extract useful time features
df['Hour'] = df['Arrival_Time'].dt.hour
df['Weekday'] = df['Arrival_Time'].dt.day_name()

# Group by hour to find busiest times
busiest_hours = df.groupby('Hour')['Passenger_Count'].sum().sort_values(ascending=False)

print("\n⏰ Busiest Hours:")
print(busiest_hours)

# Plot busiest hours
plt.figure(figsize=(10, 6))
sns.barplot(x=busiest_hours.index, y=busiest_hours.values, palette='magma')
plt.title("Busiest Hours by Passenger Count")
plt.xlabel("Hour of Day")
plt.ylabel("Total Passengers")
plt.tight_layout()
plt.savefig("plots/busiest_hours.png")
plt.show()

# Find inefficient routes
route_summary = df.groupby('Route_ID')['Passenger_Count'].sum().sort_values()
print("\n🚍 Routes with Lowest Passenger Count:")
print(route_summary.head())
