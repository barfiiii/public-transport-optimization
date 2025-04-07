import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create plots folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load CSV
df = pd.read_csv("data/city_transport.csv")

# Show preview
print("📄 Data Preview:")
print(df.head())

# Convert columns to datetime
df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
df['Departure_Time'] = pd.to_datetime(df['Departure_Time'])

# Extract time-based features
df['Hour'] = df['Arrival_Time'].dt.hour
df['Weekday'] = df['Arrival_Time'].dt.day_name()
df['Weekday_Num'] = df['Arrival_Time'].dt.weekday

# Group by hour
busiest_hours = df.groupby('Hour')['Passenger_Count'].sum().sort_values(ascending=False)
print("\n⏰ Busiest Hours:")
print(busiest_hours)

# Plot: Busiest Hours
plt.figure(figsize=(10, 6))
sns.barplot(x=busiest_hours.index, y=busiest_hours.values, hue=busiest_hours.index, palette='magma', legend=False)
plt.title("Busiest Hours by Passenger Count")
plt.xlabel("Hour of Day")
plt.ylabel("Total Passengers")
plt.tight_layout()
plt.savefig("plots/busiest_hours.png")
plt.close()

# Group by route
route_summary = df.groupby('Route_ID')['Passenger_Count'].sum().sort_values()
print("\n🚍 Routes with Lowest Passenger Count:")
print(route_summary.head())

# Plot: Least Used Routes
plt.figure(figsize=(10, 6))
sns.barplot(x=route_summary.head(10).index, y=route_summary.head(10).values, hue=route_summary.head(10).index, palette='coolwarm', legend=False)
plt.title("Least Used Routes")
plt.xlabel("Route ID")
plt.ylabel("Total Passengers")
plt.tight_layout()
plt.savefig("plots/least_used_routes.png")
plt.close()

# Plot: Passenger Count by Weekday
plt.figure(figsize=(10, 6))
sns.boxplot(x="Weekday", y="Passenger_Count", data=df, palette="viridis", order=[
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Passenger Distribution by Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.savefig("plots/passenger_by_weekday.png")
plt.close()

# ---------- Machine Learning ----------
if len(df) >= 10:
    features = df[['Route_ID', 'Hour', 'Weekday_Num']]
    target = df['Passenger_Count']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n🤖 ML Model Performance:\nMSE: {mse:.2f}\nR² Score: {r2:.2f}")

    # Example Prediction
    example = pd.DataFrame({
        'Route_ID': [3],
        'Hour': [17],
        'Weekday_Num': [4]
    })
    example_pred = model.predict(example)[0]
else:
    mse = r2 = example_pred = None
    print("\n⚠️ Not enough data to train ML model.")

# ---------- Save Summary ----------
with open("summary.txt", "w", encoding='utf-8') as file:
    file.write("📝 Transport Analysis Summary\n")
    file.write("-" * 40 + "\n")
    file.write(f"Top 3 Busiest Hours:\n{busiest_hours.head(3).to_string()}\n\n")
    file.write(f"Least Used Routes:\n{route_summary.head(3).to_string()}\n\n")
    if mse is not None:
        file.write("📈 ML Model Stats:\n")
        file.write(f"MSE: {mse:.2f}\nR² Score: {r2:.2f}\n")
        file.write(f"\n🔮 Example Prediction:\nRoute 3, 5 PM Friday → {int(example_pred)} passengers\n")
    else:
        file.write("⚠️ Not enough data to train a machine learning model.\n")

print("\n✅ Analysis complete. Check the 'plots' folder and 'summary.txt'.")
