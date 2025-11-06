
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(df_trans_historical):
    # Train a Random Forest model on historical weather data.
    # Returns trained model
    features = ['rel_humidity', 'precip', 'pressure', 'wind_speed', 'wind_direction']
    target = 'temp'

    df = df_trans_historical.sort_values("date")
    X_train = df[features]
    y_train = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_current(model, df_trans_current):
    # Use trained model to predict current temperature from a single or few records.
    # Returns prediction(s)
    features = ['rel_humidity', 'precip', 'pressure', 'wind_speed', 'wind_direction']

    # Make sure input is DataFrame with expected columns
    if isinstance(df_trans_current, pd.Series):
        df_trans_current = df_trans_current.to_frame().T

    prediction = model.predict(df_trans_current[features])
    return prediction
