import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Train a Random Forest Classifier for failure prediction.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    n_estimators : int
        Number of trees in the forest
    max_depth : int
        Maximum depth of trees
    
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model
    """
    # Initialize the model with parameters
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    return rf_model

def train_isolation_forest(X_train, contamination=0.1, n_estimators=100):
    """
    Train an Isolation Forest for anomaly detection.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    contamination : float
        Expected proportion of outliers in the data
    n_estimators : int
        Number of base estimators
    
    Returns:
    --------
    sklearn.ensemble.IsolationForest
        Trained Isolation Forest model
    """
    # Initialize the model with parameters
    if_model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    if_model.fit(X_train)
    
    return if_model

def train_time_series_model(data, feature='cpu_usage_percent'):
    """
    Train an ARIMA time series model for forecasting.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with timestamp and the target feature
    feature : str
        Feature to forecast
    
    Returns:
    --------
    dict
        Dictionary containing the trained model and metadata
    """
    # Ensure data has timestamp column
    if 'timestamp' not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column for time series analysis")
    
    # Ensure the feature exists
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found in data")
    
    # Prepare time series data
    ts_data = data.sort_values('timestamp')
    
    # Set timestamp as index and select only the target feature
    ts_series = ts_data.set_index('timestamp')[feature]
    
    # Try to find the best ARIMA parameters using auto_arima
    try:
        # Import pmdarima if available
        from pmdarima import auto_arima
        
        # Find the best parameters
        auto_model = auto_arima(
            ts_series,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            m=1,  # No seasonality
            d=None,  # Let auto_arima determine the differencing
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        # Extract the best parameters
        p, d, q = auto_model.order
        
    except ImportError:
        # If pmdarima is not available, use default parameters
        p, d, q = 1, 1, 1
    
    # Train ARIMA model
    model = ARIMA(ts_series, order=(p, d, q))
    fitted_model = model.fit()
    
    # Forecast for the next 10 periods
    forecast_periods = 10
    forecast = fitted_model.forecast(steps=forecast_periods)
    
    # Prepare result
    result = {
        'model': fitted_model,
        'forecast': forecast,
        'feature': feature,
        'last_timestamp': ts_series.index[-1],
        'order': (p, d, q)
    }
    
    return result

def train_arima_model(time_series, order=(5, 1, 0)):
    """
    Train an ARIMA model on a time series.
    
    Parameters:
    -----------
    time_series : pandas.Series
        Time series data with datetime index
    order : tuple
        ARIMA order (p, d, q)
    
    Returns:
    --------
    statsmodels.tsa.arima.model.ARIMAResults
        Fitted ARIMA model
    """
    model = ARIMA(time_series, order=order)
    fitted_model = model.fit()
    return fitted_model
