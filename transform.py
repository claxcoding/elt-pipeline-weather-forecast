
import pandas as pd
from google.cloud import bigquery

def transform_daily_data(df):
    df['date'] = pd.to_datetime(df['time']).dt.date
    df = df.rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precip_total"
    })
    df = df.dropna(subset=['temp_max', 'temp_min', 'precip_total'])
    return df[['date', 'temp_max', 'temp_min', 'precip_total']]
