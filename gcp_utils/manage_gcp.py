
from create_dataset import create_dataset
from create_table import create_table
from load_config import load_config

def main(project, dataset, table):

    # Load the config from the file we just created
    config = load_config('/content/drive/MyDrive/elt_pipeline_weather_forecast/config.yaml')

    # Access the configurations
    project_id = config['project']['project_id']
    dataset_historical = config['project']['dataset_historical']
    table_historical = config['project']['table_historical']
    dataset_month = config['project']['dataset_month']
    table_8h = config['project']['table_8h']
    dataset_prediction = config['project']['dataset_prediction']
    params_current = config['params_current']
    params_historical = config['params_historical']
    url_historical_2017_2014 = config['api_historical']
    url_historical_2017 = config['api_historical']['elt_weather_table_historical_siegburg_2017']
    url_current = config['api_current']['elt_weather_table_current']

    create_dataset(project, dataset)
    create_table(project, dataset, table)

if __name__ == "__main__":
    # You can provide default values here, or leave it empty
    main("project_id", "dataset", "table")

