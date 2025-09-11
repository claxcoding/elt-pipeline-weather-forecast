
from gcp_utils.create_dataset import create_dataset
from gcp_utils.create_table import create_table

def main(project, dataset, table):
    create_dataset(project, dataset)
    create_table(project, dataset, table)

if __name__ == "__main__":
    # You can provide default values here, or leave it empty
    main("etl-weather-467607", "default_dataset", "default_table")
