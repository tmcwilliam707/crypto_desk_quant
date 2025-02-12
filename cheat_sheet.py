import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic financial data
def generate_data(n=500):
    np.random.seed(42)
    X = np.random.randn(n, 5)
    y = (np.random.rand(n) > 0.5).astype(int)  # Binary classification for ML models
    time_series = np.cumsum(np.random.randn(n))  # Simulated time series
    return X, y, time_series

X, y, time_series = generate_data()

# Split data for ML models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Statistical Model: OLS Regression ---
X_ols = sm.add_constant(X)
model_ols = sm.OLS(y, X_ols).fit()
print("OLS Summary:")
print(model_ols.summary())

# --- Statistical Model: ARIMA for Time Series Forecasting ---
model_arima = ARIMA(time_series, order=(1,1,1))
model_arima_fit = model_arima.fit()
print("ARIMA Summary:")
print(model_arima_fit.summary())

# --- Machine Learning Model: Logistic Regression ---
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print("Logistic Regression Score:", log_reg.score(X_test, y_test))

# --- Machine Learning Model: Support Vector Machine (SVM) ---
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
print("SVM Score:", svm_model.score(X_test, y_test))

# --- Machine Learning Model: Neural Network (MLP) ---
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=500)
mlp.fit(X_train, y_train)
print("Neural Network Prediction:", mlp.predict(X_test[:5]))

# --- Machine Learning Model: Gradient Boosting ---
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)
print("Gradient Boosting Score:", gbm.score(X_test, y_test))