
from google.cloud import bigquery

def create_dataset(PROJECT_ID, dataset_id, location="EU"):
    # Creates a BigQuery dataset if it doesn't exist.
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset '{dataset_id}' already exists.")
    except:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        print(f"Dataset '{dataset_id}' created.")
