
import pandas as pd
from google.cloud import bigquery

def transform_data(df):
    df['date'] = pd.to_datetime(df['time']).dt.date
    df = df.rename(columns={
        "temperature_2m": "temp",
        "relative_humidity_2m": "rel_humidity",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "precipitation": "precip",
    })
    df = df.dropna(subset=['temp', 'rel_humidity', 'pressure',
                          'wind_speed', 'wind_direction', 'precip'])
    return df[['date', 'temp', 'rel_humidity', 'pressure',
                          'wind_speed', 'wind_direction', 'precip']]
