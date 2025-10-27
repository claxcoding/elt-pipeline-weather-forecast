
import matplotlib.pyplot as plt

def visualize_current_prediction(df_trans_current, pred_temp):

    # Display current input features and predicted temperature.
    current = df_trans_current.iloc[0]  # Assuming 1 row
    print("Current weather input:")
    display(current.to_frame(name="value"))
    print(f"Predicted temperature: {pred_temp[0]:.2f} °C")

    # Plot weather features + predicted temp
    feature_values = current[['rel_humidity', 'precip', 'pressure', 'wind_speed', 'wind_direction']]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(feature_values.index, feature_values.values, color='skyblue')
    plt.title("Current Weather Features")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.figtext(0.99, 0.01, 'Weather data provided by Open-Meteo.com (CC BY 4.0)', ha='right', va='bottom', fontsize=8, color='gray')
    plt.show()

    # Show predicted temperature as a separate bar
    plt.figure(figsize=(4, 5))
    plt.bar(['Predicted Temp'], [pred_temp[0]], color='salmon')
    plt.ylim(0, max(50, pred_temp[0] + 5))  # Adjust Y-axis
    plt.title("Predicted Temperature")
    plt.ylabel("°C")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_predictions_over_time(dates, actual, predicted):

    # Plot predicted vs actual temperature over time.
    plt.figure(figsize=(12,6))
    plt.plot(dates, actual, label='Actual Temperature', linewidth=2)
    plt.plot(dates, predicted, label='Predicted Temperature', linestyle='--')
    plt.title("Predicted vs Actual Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.99, 0.01, 'Weather data provided by Open-Meteo.com (CC BY 4.0)', ha='right', va='bottom', fontsize=8, color='gray')
    plt.show()
