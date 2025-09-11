
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model(df):
    X = df[['temp_min', 'precip_total']]
    y = df['temp_max']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression().fit(X_train, y_train)
    return model, X_test, y_test
