
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

def query_bigquery(project_id, dataset, table, columns="*", filters=None, order_by=None):

    # Build a BigQuery SQL SELECT statement.

    # Parameters:
    # - project_id: GCP project id
    # - dataset: BigQuery dataset name
    # - table: Table name
    # - columns: String or list of columns to select
    # - filters: list of filter strings, e.g. ["EXTRACT(MONTH FROM date) = 10", "temperature > 20"]
    # - order_by: string of column(s) to order by

    # Returns:
    # - Full SQL query string

    # If columns is a list, join by comma, else use as-is
    if isinstance(columns, list):
        columns_str = ", ".join(columns)
    else:
        columns_str = columns

    query = f"SELECT {columns_str} FROM `{project_id}.{dataset}.{table}`"

    # Build WHERE clause if filters exist
    if filters:
        if isinstance(filters, list):
            filters_str = " AND ".join(filters)
        else:
            filters_str = filters
        query += f" WHERE {filters_str}"

    if order_by:
        query += f" ORDER BY {order_by}"

    return query


def extract_with_query(project_id, dataset, table, columns="*", filters=None, order_by=None):
    '''Flexible extract using custom query.'''
    client = bigquery.Client(project=project_id)
    query = query_bigquery(project_id, dataset, table, columns, filters, order_by)
    df = client.query(query).to_dataframe()
    return df
