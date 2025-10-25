
import requests
import pandas as pd
from load_config import load_config

def api_fetch(url, params=None):
    # Fetch API data with optional query parameters.
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def api_fetch_url(url, params=None):

    response = requests.get(url, params=params)
    response.raise_for_status()
    api_data = response.json()

    return api_data

def fetch_historical_weather():

    # Fetch historical weather data for all years defined in config.yaml.
    config = load_config('/content/drive/MyDrive/elt_pipeline_weather_forecast/config.yaml')

    historical_data_dict = {}  # store raw JSON responses per year
    historical_df_dict = {}    # store converted DataFrames per year

    # Loop through all URLs in config['api_historical']
    for key, url in config["api_historical"].items():
        year = key.split("_")[-1]  # extract year, e.g. "2017"
        print(f"Fetching weather data for {year}...")

        # Fetch from API
        data = api_fetch_url(url)
        historical_data_dict[year] = data

        # Convert the hourly data part to a DataFrame
        hourly_data = data.get("hourly")
        if hourly_data:
            df = pd.DataFrame(hourly_data)
            df["year"] = int(year)
            historical_df_dict[year] = df
            print(f"  → Loaded {len(df)} hourly records for {year}")

    return historical_data_dict, historical_df_dict
