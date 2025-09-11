
from config import get_config
from extract import extract_daily_data, extract_with_query
from transform import transform_daily_data
from train import train_model
from predict import predict
from visualize import plot_predictions
from load import load_to_bigquery
from query_utils import build_query, prompt_user_for_query


def main():
    config = get_config()
    project = config["project_id"]
    dataset = config["dataset"]
    table = config["table_daily"]

    # Option: Use user-defined query
    use_prompt = input("Use custom query? (yes/no): ").strip().lower() == "yes"

    if use_prompt:
        query_params = prompt_user_for_query()
        query = build_query(
            project=project,
            dataset=dataset,
            table=table,
            columns=query_params["columns"],
            filters=query_params["filters"],
            order_by=query_params["order_by"]
        )
        df_raw = extract_with_query(project, query)
    else:
        # Use config-driven query or default full-table read
        custom_query = config.get("custom_query_config", None)

        if custom_query:
            df_raw = extract_with_query(
                project,
                build_query(
                    project,
                    dataset,
                    table,
                    columns=custom_query.get("columns", "*"),
                    filters=custom_query.get("filters"),
                    order_by=custom_query.get("order_by")
                )
            )
        else:
            df_raw = extract_daily_data(project, dataset, table)

    # Continue ETL
    df_trans = transform_daily_data(df_raw)
    model, X_test, y_test = train_model(df_trans)
    df_pred = predict(model, X_test, y_test, df_trans)
    plot_predictions(df_pred)
    load_to_bigquery(df_pred, project, config["dataset_pred"], config["table_pred"])

if __name__ == "__main__":
    main()

