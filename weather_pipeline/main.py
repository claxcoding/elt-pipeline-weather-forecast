
import yaml
import json
import pandas as pd
import numpy as np
from load_config import load_config
from fetch_utils import api_fetch, api_fetch_url, fetch_historical_weather
from query_utils import extract_weather_current, extract_full_query
from load_utils import load_to_bigquery_raw, load_to_bigquery, numpy_to_dataframe, current_month
from transform_utils import transform_data
from predictor_utils import train_model, predict_current
from sklearn.metrics import mean_absolute_error
from transform_utils import transform_data
from visualization_utils import visualize_current_prediction, plot_predictions_over_time


def main(project_id, dataset_historical, table_historical, dataset_month, table_id_8h_current, table_id_8h_prediction_current, dataset_prediction):
    print(">>> main() is running.")

    # Load the config from the .yaml file
    config = load_config('/content/drive/MyDrive/elt_pipeline_weather_forecast/config.yaml')

    # Access the configurations
    project_id = config['project']['project_id']
    dataset_historical = config['project']['dataset_historical']
    table_historical = config['project']['table_historical']
    dataset_month = config['project']['dataset_month']
    table_8h = config['project']['table_8h']
    dataset_prediction = config['project']['dataset_prediction']
    params_current = config['params_current']
    params_historical = config['params_historical']
    url_historical_2017_2014 = config['api_historical']
    url_historical_2017 = config['api_historical']['elt_weather_table_historical_siegburg_2017']
    url_current = config['api_current']['elt_weather_table_current']

    # Print values to confirm
    print(f"Project ID: {project_id}")
    print(f"Dataset Historical: {dataset_historical}")
    print(f"Table Historical: {table_historical}")
    print(f"Dataset Month: {dataset_month}")
    print(f"Table Current: {table_id_8h_current}")

    # Fetching data from the API
    print(">>> Starting main API fetch")
    # Fetch historical data from the weather API
    weather_data_historical = api_fetch_url(url_historical_2017)

    # Fetch historical data from the weather API for each year 2017-2024
    historical_data_dict, historical_df_dict = fetch_historical_weather()

    # Example: access one year’s (2017) data
    weather_data_historical_2017 = historical_data_dict.get("2017")

    print("Received data for the year 2017:")
    print(f"Type: {type(weather_data_historical_2017)}")

    if weather_data_historical_2017:
        print("Top-level keys:", list(weather_data_historical_2017.keys()))
        hourly_data = weather_data_historical_2017.get("hourly")
        if hourly_data:
            print("Hourly keys:", list(hourly_data.keys())[:10])
            print("First 3 time entries:", hourly_data.get("time", [])[:3])
    else:
        print("No data found for 2017.")

    # Fetch current data from the weather API
    weather_data_current = api_fetch_url(url_current)
    print("API fetch for current data:",weather_data_current)
    print("Keys in weather_data_current:", weather_data_current.keys())
    print("Type of weather_data_current:", type(weather_data_current))

    # Access the nested current data for current data
    df_current = pd.DataFrame([weather_data_current['current']])
    print("Current data:")
    print(df_current.head())

    print(">>> API fetch ends")

    print(">>> Loading fetched historical data to BigQuery")

    # Load fetched historical data to BigQuery
    for year, df in historical_df_dict.items():
        if df.empty:
            print(f"No data for year {year}, skipping.")
            continue
        table_name = f"elt_weather_table_historical_siegburg_{year}"
        print(f"Loading data for year {year} into {table_name}...")

        load_to_bigquery_raw(df, project_id, dataset_historical, table_name)

    # Load fetched current data to BigQuery
    print(">>> Loading fetched current data to BigQuery")
    load_to_bigquery_raw(df_current, project_id, dataset_month, table_id_8h_current)
    print(">>> Loading ends")

    # # Get the current month (for future seasonal predictions)
    # month_input = current_month()
    # print(f"Current month: {month_input}")

    # # Extract historical weather data from BigQuery and format to DataFrame (for future seasonal predictions)
    # print(f">>> Extracting historical data from BigQuery for the month {month_input}")

    # Load historical data from BigQuery for the current month
    # month_input = 1  # Example: January (for future seasonal predictions)

    df_historical = []

    # The following code loops through yearly historical BigQuery tables, runs a SELECT that pulls several weather columns and computes 1-step past values (LAG)
    # and a 1-step future target (LEAD), collects the returned pandas DataFrames, concatenates them into one large DataFrame,
    # and drops rows with any missing values (which removes rows with NULL lag/lead)

    for year in range(2017, 2025):
        table = f"elt_weather_table_historical_siegburg_{year}"
        query = f'''
        SELECT
            time,
            temperature_2m AS temp,
            relative_humidity_2m AS rel_humidity,
            surface_pressure AS pressure,
            wind_speed_10m AS wind_speed,
            wind_direction_10m AS wind_direction,
            precipitation AS precip,

            # These lines use SQL window functions — LAG(column, 1) OVER (ORDER BY time) returns the
            # column value from the previous row when rows are ordered by time.
            # Each LAG(...,1) gives the immediate past observation for that metric

            LAG(temperature_2m, 1) OVER (ORDER BY time) AS lag_temp_1,
            LAG(relative_humidity_2m, 1) OVER (ORDER BY time) AS lag_rel_humidity_1,
            LAG(surface_pressure, 1) OVER (ORDER BY time) AS lag_pressure_1,
            LAG(wind_speed_10m, 1) OVER (ORDER BY time) AS lag_wind_speed_1,
            LAG(wind_direction_10m, 1) OVER (ORDER BY time) AS lag_wind_direction_1,
            LAG(precipitation, 1) OVER (ORDER BY time) AS lag_precip_1,

            LEAD(temperature_2m, 1) OVER (ORDER BY time) AS temp_future

        FROM `{project_id}.{dataset_historical}.{table}`
        '''

        df = extract_full_query(query, project_id)
        if not df.empty:
            df_historical.append(df)
        else:
            print(f"No data found for year {year}.")

    if df_historical:
        df_historical = pd.concat(df_historical, ignore_index=True)
        df_historical.dropna(inplace=True)  # Remove rows where lag/lead = NULL

    print(df_historical.head())
    print(f"Total rows historical data: {len(df_historical)}")

    # Extract current weather data from BigQuery and format into DataFrame
    print(">>> Extracting current data from BigQuery")
    # Extract current weather data from BigQuery and format into DataFrame
    df_current = extract_weather_current(project_id, dataset_month, table_id_8h_current)
    print("Current data from BigQuery:")
    print(df_current.head())
    print(">>> Extracting ends")

    # Transform historical weather data
    print(">>> Transforming historical data")
    df_trans_historical = transform_data(df_historical)
    print(df_trans_historical.head())

    # Transform current weather data
    print(">>> Transforming current data")
    df_trans_current = transform_data(df_current)
    print(df_trans_current.head())
    print(">>> Transforming ends")

    print(">>> Machine Learning prediction and visualization starts")
    # Train ML model and make predictions
    # df_trans_historical → large set
    # df_trans_current → 1 row from current API

    # Train model on past data
    model = train_model(df_trans_historical)
    # Predict temp from current values
    pred_temp = predict_current(model, df_trans_current)
    print(f"→ Predicted current temperature: {pred_temp[0]:.2f} °C")

    X_test = df_trans_historical[['rel_humidity', 'precip', 'pressure', 'wind_speed', 'wind_direction']]
    y_test = df_trans_historical['temp']
    y_pred = model.predict(X_test)
    plot_predictions_over_time(df_trans_historical['date'], y_test, y_pred)

    # Show prediction nicely
    visualize_current_prediction(df_trans_current, pred_temp)

    print(">>> Machine Learning prediction and visualization end")

    # Load prediction to BigQuery
    print(">>> Loading prediction to BigQuery")

    df_pred = numpy_to_dataframe(pred_temp)
    load_to_bigquery(df_pred, project_id, dataset_prediction, table_id_8h_prediction_current)

    print(">>> Loading ends")

    print(">>> main() is done.")

if __name__ == "__main__":
    # You can provide default values here, or leave it empty
    pass # This will not run when imported
