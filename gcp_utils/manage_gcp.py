
from create_dataset import create_dataset
from create_table import create_table
from load_config import load_config

def main(project, dataset, table):

    # Load the config from the file we just created
    config = load_config('/content/drive/MyDrive/elt_pipeline_weather_forecast/config.yaml')

    # Create datasets and tables
    create_dataset(project, dataset)
    create_table(project, dataset, table)

if __name__ == "__main__":
    # You can provide default values here, or leave it empty
    main("project_id", "dataset", "table")
    