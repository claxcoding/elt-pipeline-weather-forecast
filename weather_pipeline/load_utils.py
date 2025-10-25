

import time
import pandas as pd
import numpy as np
from google.cloud import bigquery

def load_to_bigquery_raw(data, project_id, dataset, table):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table}"

    try:
        # Check if table exists
        table_obj = client.get_table(table_ref)

        # Query row count
        row_count = table_obj.num_rows
        if row_count > 0:
            print(f"Table {table_ref} already has data ({row_count} rows). Skipping load.")
            return
        else:
            print(f"Table {table_ref} exists but is empty. Loading data...")
    except Exception as e:
        # Table doesn't exist
        print(f"Table {table_ref} does not exist. It will be created and data will be loaded.")

    # Load data
    job = client.load_table_from_dataframe(data, table_ref)
    job.result()
    print(f"Loaded {job.output_rows} rows to {table_ref}")

def load_to_bigquery(data, project_id, dataset, table):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table}"

    try:
        # Check if table exists
        table_obj = client.get_table(table_ref)

        # Query row count
        row_count = table_obj.num_rows
        if row_count > 0:
            print(f"Table {table_ref} already has data ({row_count} rows). Skipping load.")
            return
        else:
            print(f"Table {table_ref} exists but is empty. Loading data...")
    except Exception as e:
        # Table doesn't exist
        print(f"Table {table_ref} does not exist. It will be created and data will be loaded.")

    # Load data
    job = client.load_table_from_dataframe(data, table_ref)
    job.result()
    print(f"Loaded {job.output_rows} rows to {table_ref}")

# Function to convert NumPy array to DataFrame
def numpy_to_dataframe(np_array):
    # If np_array is a single value (scalar), reshape it into a DataFrame
    if np_array.ndim == 1:  # For a 1D array
        return pd.DataFrame(np_array, columns=["prediction"])
    elif np_array.ndim == 2:  # For a 2D array
        return pd.DataFrame(np_array, columns=["prediction"])
    else:
        raise ValueError("Only 1D or 2D NumPy arrays are supported.")
