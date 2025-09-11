
import pandas as pd
from google.cloud import bigquery
from query_utils import query_bigquery

def extract_daily_data(project_id, dataset, table_daily):
    '''Default full extract.'''
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table_daily}"
    query = f"SELECT * FROM `{table_ref}`"
    df = client.query(query).to_dataframe()
    return df

def extract_hourly_data(project_id, dataset, table_daily):
    '''Default full extract.'''
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table_daily}"
    query = f"SELECT * FROM `{table_ref}`"
    df = client.query(query).to_dataframe()
    return df

def extract_with_query(project_id, dataset, table, columns="*", filters=None, order_by=None):
    '''Flexible extract using custom query.'''
    client = bigquery.Client(project=project_id)
    query = query_bigquery(project_id, dataset, table, columns, filters, order_by)
    df = client.query(query).to_dataframe()
    return df
