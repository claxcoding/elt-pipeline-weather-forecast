
import pandas as pd

def predict(model, X_test, y_test, df_transformed):
    y_pred = model.predict(X_test)
    return pd.DataFrame({
        'date': df_transformed.loc[X_test.index, 'date'],
        'temp_max': y_test,
        'pred_temp_max': y_pred
    })
