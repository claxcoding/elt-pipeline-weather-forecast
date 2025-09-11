
# Weather Forecasting ETL Pipeline

This project demonstrates an end-to-end ETL (Extract, Transform, Load) pipeline built in Python using a Jupyter/Colab notebook.
It extracts weather data, transforms it, trains a simple linear regression model to predict max temperatures, and loads results into BigQuery.

## Project Overview

Brief description of the dataset and problem.

## ETL Process

1. **Extract**: [fetch from API -> load into Google Cloud via BigQuery -> SQL queries]
2. **Transform**: [Data cleaning, feature engineering]
3. **Load**: [load ML results back into Google Cloud via BigQuery]

## Files

- `etl_pipeline_weather.ipynb` – Main notebook for .py files and runner scripts
- `gcp_utils/` – Moduled .py files for GCP utilization
- `weather_pipeline/` – Moduled .py files for the ETL/ML pipeline
- `requirements.txt` – List of required packages

## Modules gcp_utils/

- `create_dataset.py`: Create dataset in Google Cloud
- `create_table.py`: Create Table in Google Cloud inside a dataset
- `jobtables.py`: Create job for BigQuery
- `manage_gcp.py`: GCP utilization runner

## Modules weather_pipeline/

- `config.py`: Configuration settings
- `extract.py`: Extract from BigQuery
- `transform.py`: Data cleaning
- `train.py`: Linear regression model
- `predictor.py`: Predictions
- `visualize.py`: Plots
- `load.py`: Load predictions to BigQuery
- `query_utils.py`: Utility functions for BigQuery queries
- `requirements.txt`: Dependencies
- `main.py`: Pipeline runner

## Requirements

- Google Cloud credentials
- Python 3.7+

## Future Work

- Dockerize (not functionable yet)
- Integrate/orchestrate with Airflow/dbt 

