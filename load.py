
import pandas as pd
from google.cloud import bigquery

def load_to_bigquery(df, project_id, dataset, table):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table}"
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    print(f"Loaded {job.output_rows} rows to {table_ref}")
