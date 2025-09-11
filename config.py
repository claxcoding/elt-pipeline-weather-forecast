
def get_config():
    return {
        "project_id": "etl-weather-467607",
        "dataset": "etl_weather_dataset_siegburg_2024_hourly_daily",
        "table_daily": "daily_weather_siegburg",
        "dataset_pred": "etl_weather_dataset_siegburg_2024_predictions",
        "table_pred": "siegburg_pred_lin_regression_2024-2025",

        # Optional SQL query config
        "custom_query_config": {
            "columns": "time, temperature_2m_max, temperature_2m_min, precipitation_sum",
            "filters": "temperature_2m_max IS NOT NULL",
            "order_by": "time DESC"
        }
    }
