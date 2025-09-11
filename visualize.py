
import matplotlib.pyplot as plt

def plot_predictions(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['date'], df['temp_max'], label='Actual')
    plt.scatter(df['date'], df['pred_temp_max'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
