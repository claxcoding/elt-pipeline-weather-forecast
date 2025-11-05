
# prefect pipeline
import pandas as pd
import numpy as np
import datetime
from prefect import flow, task
from load_config import load_config
from load_utils import load_to_bigquery, load_to_bigquery_raw, numpy_to_dataframe
from fetch_utils import api_fetch_url, fetch_historical_weather
from transform_utils import transform_data
from predictor_utils import train_model, predict_current
from visualization_utils import plot_predictions_over_time, visualize_current_prediction
from query_utils import extract_full_query, extract_weather_current
from gcp_utils.create_dataset import create_dataset
from gcp_utils.create_table import create_table

# -------------
# Prefect Tasks
# -------------

@task
def load_config_task(config_path: str):
    return load_config(config_path)


@task(retries=3, retry_delay_seconds=10)
def create_dataset_task(project_id: str, dataset_id: str):
    return create_dataset(project_id, dataset_id)


@task(retries=3, retry_delay_seconds=10)
def create_table_task(project_id: str, dataset_id: str, table_id: str):
    return create_table(project_id, dataset_id, table_id)


@task
def fetch_historical_task(config: dict):
    historical_data_dict, historical_df_dict = fetch_historical_weather()
    return historical_data_dict, historical_df_dict


@task
def fetch_current_task(url_current: str):
    return api_fetch_url(url_current)


@task
def load_historical_to_bq(df_dict: dict, project_id: str, dataset_historical: str):
    for year, df in df_dict.items():
        if df.empty:
            continue
        table_name = f"elt_weather_table_historical_siegburg_{year}"
        load_to_bigquery_raw(df, project_id, dataset_historical, table_name)


@task
def load_current_to_bq(df_current: pd.DataFrame, project_id: str, dataset_month: str, table_current: str):
    load_to_bigquery_raw(df_current, project_id, dataset_month, table_current)


@task
def get_seasonal_historical(project_id: str, dataset_historical: str):
    current_time = datetime.datetime.now()
    target_month = current_time.month
    season_months = [(target_month - 1) % 12 or 12,
                     target_month,
                     (target_month + 1 - 1) % 12 + 1]

    df_historical_list = []

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

          -- Lags
          LAG(temperature_2m, 1) OVER (ORDER BY time) AS lag_temp_1h,
          LAG(temperature_2m, 2) OVER (ORDER BY time) AS lag_temp_2h,
          LAG(temperature_2m, 3) OVER (ORDER BY time) AS lag_temp_3h,

          LAG(relative_humidity_2m, 1) OVER (ORDER BY time) AS lag_rel_humidity_1h,
          LAG(relative_humidity_2m, 2) OVER (ORDER BY time) AS lag_rel_humidity_2h,
          LAG(relative_humidity_2m, 3) OVER (ORDER BY time) AS lag_rel_humidity_3h,

          LAG(surface_pressure, 1) OVER (ORDER BY time) AS lag_pressure_1h,
          LAG(surface_pressure, 2) OVER (ORDER BY time) AS lag_pressure_2h,
          LAG(surface_pressure, 3) OVER (ORDER BY time) AS lag_pressure_3h,

          LAG(wind_speed_10m, 1) OVER (ORDER BY time) AS lag_wind_speed_1h,
          LAG(wind_speed_10m, 2) OVER (ORDER BY time) AS lag_wind_speed_2h,
          LAG(wind_speed_10m, 3) OVER (ORDER BY time) AS lag_wind_speed_3h,

          LAG(wind_direction_10m, 1) OVER (ORDER BY time) AS lag_wind_direction_1h,
          LAG(wind_direction_10m, 2) OVER (ORDER BY time) AS lag_wind_direction_2h,
          LAG(wind_direction_10m, 3) OVER (ORDER BY time) AS lag_wind_direction_3h,

          LAG(precipitation, 1) OVER (ORDER BY time) AS lag_precip_1h,
          LAG(precipitation, 2) OVER (ORDER BY time) AS lag_precip_2h,
          LAG(precipitation, 3) OVER (ORDER BY time) AS lag_precip_3h
        FROM `{project_id}.{dataset_historical}.{table}`
        '''
        
        df = extract_full_query(query, project_id)

        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df['month'] = df['time'].dt.month

            # Filter data to ±1 month around current time
            df = df[df['month'].isin(season_months)]

            # Add wind direction cyclical encoding
            df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_direction']))
            df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_direction']))

            # Add diurnal cyclical encodings
            df['hour'] = df['time'].dt.hour
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Drop rows with missing lag values
            df.dropna(inplace=True)
            df_historical_list.append(df)
            print(f"Loaded {len(df)} rows from {table}")
        else:
            print(f"No data found for year {year}.")

    # Combine and summarize 
    if df_historical_list:
        df_historical = pd.concat(df_historical_list, ignore_index=True)
        print(f"Total rows after seasonal concatenation: {len(df_historical)}")

        # Optional rolling/aggregate features
        df_historical['temp_mean_last3h'] = df_historical[['lag_temp_1h','lag_temp_2h','lag_temp_3h']].mean(axis=1)
        df_historical['wind_speed_mean_last3h'] = df_historical[['lag_wind_speed_1h','lag_wind_speed_2h','lag_wind_speed_3h']].mean(axis=1)
        df_historical['humidity_mean_last3h'] = df_historical[['lag_rel_humidity_1h','lag_rel_humidity_2h','lag_rel_humidity_3h']].mean(axis=1)
        
        print(df_historical.head())       
        return df_historical
    else:
        print("No seasonal historical data loaded.")


@task
def extract_current_from_bq(project_id: str, dataset_month: str, table_current: str):
    return extract_weather_current(project_id, dataset_month, table_current)


@task
def train_and_predict(df_historical: pd.DataFrame, df_current: pd.DataFrame):
    df_trans_historical = transform_data(df_historical)
    df_trans_current = transform_data(df_current)
    model = train_model(df_trans_historical)
    pred_temp = predict_current(model, df_trans_current)

    X_test = df_trans_historical[['rel_humidity', 'precip', 'pressure', 'wind_speed', 'wind_direction']]
    y_test = df_trans_historical['temp']
    y_pred = model.predict(X_test)
    plot_predictions_over_time(df_trans_historical['date'], y_test, y_pred)
    visualize_current_prediction(df_trans_current, pred_temp)
    return pred_temp


@task
def load_prediction_to_bq(pred_temp: np.ndarray, project_id: str, dataset_prediction: str, table_pred: str):
    df_pred = numpy_to_dataframe(pred_temp)
    load_to_bigquery(df_pred, project_id, dataset_prediction, table_pred)


# ------------------
# Flow Orchestration
# ------------------

@flow(name="Weather ELT + Machine Learning Pipeline")
def weather_pipeline_flow(config_path: str):
    print("Running weather pipeline...")

    # Load config
    config = load_config_task(config_path)

    project_id = config['project']['project_id']
    dataset_historical = config['project']['dataset_historical']
    dataset_month = config['project']['dataset_month']
    url_historical_2017_2014 = config['api_historical']
    url_current = config['api_current']['elt_weather_table_current']
    dataset_prediction = config['project']['dataset_prediction']

    # Get timestamps
    timestamp_bq = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    table_current = f"elt_weather_8h_table_{timestamp_bq}"
    table_prediction = f"elt_weather_prediction_table_{timestamp_bq}"

    # Create datasets and tables
    # Loop through each table name in the list 'url_historical_2017_2014'
    # This creates tables for each historical weather table in the BigQuery dataset.
    for n in url_historical_2017_2014:
      create_dataset_task(project_id, dataset_historical)
      create_table_task(project_id, dataset_historical, n)

    create_dataset_task(project_id, dataset_month)
    create_table_task(project_id, dataset_month, table_current)

    create_dataset_task(project_id, dataset_prediction)
    create_table_task(project_id, dataset_prediction, table_prediction)


    # Fetch and load historical & current data
    historical_data_dict, historical_df_dict = fetch_historical_task(config)
    load_historical_to_bq(historical_df_dict, project_id, dataset_historical)

    current_data = fetch_current_task(url_current)
    df_current = pd.DataFrame([current_data['current']])
    load_current_to_bq(df_current, project_id, dataset_month, table_current)

    # Seasonal historical + current from BQ
    df_historical_seasonal = get_seasonal_historical(project_id, dataset_historical)
    df_current_bq = extract_current_from_bq(project_id, dataset_month, table_current)

    # Train + Predict
    pred_temp = train_and_predict(df_historical_seasonal, df_current_bq)

    # Load prediction to BQ
    load_prediction_to_bq(pred_temp, project_id, dataset_prediction, table_prediction)

# --------
# Run Flow
# --------

if __name__ == "__main__":
    weather_pipeline_flow("/content/drive/MyDrive/elt_pipeline_weather_forecast/config.yaml")
    weather_pipeline_flow.visualize(filename="/content/drive/MyDrive/elt_weather_pipeline_dag.png")
