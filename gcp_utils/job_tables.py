
from google.cloud import bigquery

def job_tables(PROJECT_ID, dataset_id, table_id, df):
    # Create BigQuery Instance, Datasets, Tables.
    # Loads a pandas DataFrame into a BigQuery table.
    client = bigquery.Client(project=PROJECT_ID)                                # create instance of bigquery.Client class, project argument specifies the Google Cloud project ID
    table_ref = f"{PROJECT_ID}.{dataset_id}.{table_id}"
    job = client.load_table_from_dataframe(df, table_ref)                       # Start a BigQuery load job from a DataFrame
    job.result()                                                                # Wait for the load job to complete
    print(f"{job.output_rows} Zeilen nach BigQuery geladen.")                   # Print the number of rows loaded
