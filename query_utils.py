
def query_bigquery(
    project_id,
    dataset_id,
    table_id,
    columns="*",
    filters=None,
    order_by=None,
):
    '''Build a flexible SQL query for a BigQuery table.'''
    query = f"SELECT {columns} FROM `{project_id}.{dataset_id}.{table_id}`"

    if filters:
        query += f" WHERE {filters}"
    if order_by:
        query += f" ORDER BY {order_by}"

    return query

    # query_utils.py

def build_query(project, dataset, table, columns="*", filters=None, order_by=None):
    '''Builds a flexible SQL query for a BigQuery table.'''
    query = f"SELECT {columns} FROM `{project}.{dataset}.{table}`"

    if filters:
        query += f" WHERE {filters}"

    if order_by:
        query += f" ORDER BY {order_by}"

    return query

def prompt_user_for_query():
    '''Prompts the user for custom query parameters.'''
    print("Customize your BigQuery query (leave blank to skip)")

    columns = input("Columns to select (default '*'): ").strip() or "*"
    filters = input("WHERE clause (e.g., temp_max > 30): ").strip() or None
    order_by = input("ORDER BY clause (e.g., date DESC): ").strip() or None

    return {
        "columns": columns,
        "filters": filters,
        "order_by": order_by
    }

