
# Weather Forecasting ETL Pipeline

This project extracts weather data, transforms it, trains a regression model to predict max temperatures, and loads results into BigQuery.

## Modules

- `config.py`: Configuration settings
- `extract.py`: Extract from BigQuery
- `transform.py`: Data cleaning
- `train.py`: Linear regression model
- `predict.py`: Predictions
- `visualize.py`: Plots
- `load.py`: Load predictions to BigQuery
- `main.py`: Pipeline runner

## Requirements

- Google Cloud credentials
- Python 3.7+
