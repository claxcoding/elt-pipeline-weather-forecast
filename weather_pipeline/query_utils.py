
import pandas as pd
from google.cloud import bigquery

def extract_weather_historical(project_id, dataset_historical, table_historical):
    '''Default full extract.'''
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_historical}.{table_historical}"
    query = f"SELECT * FROM `{table_ref}`"
    df = client.query(query).to_dataframe()
    return df

def extract_weather_current(project_id, dataset_current, table_current):
    '''Default full extract.'''
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_current}.{table_current}"
    query = f"SELECT * FROM `{table_ref}`"
    df = client.query(query).to_dataframe()
    return df

def extract_full_query(query, project_id):
    client = bigquery.Client(project=project_id)
    try:
        df = client.query(query).to_dataframe()
        print(f"Query executed successfully. Retrieved {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

