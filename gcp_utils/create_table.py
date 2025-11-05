
from google.cloud import bigquery

def create_table(project_id, dataset_id, table_id):

    # Creates a BigQuery table if it doesn't exist.
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    try:
        client.get_table(table_ref)
        print(f"Table '{table_id}' already exists.")
    except:
        # table = bigquery.Table(table_ref, schema=schema)
        table = bigquery.Table(table_ref)
        client.create_table(table)
        print(f"Table '{table_id}' created.")
