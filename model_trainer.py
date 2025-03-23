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
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(ts_data['timestamp']):
        ts_data['timestamp'] = pd.to_datetime(ts_data['timestamp'])
    
    # Set timestamp as index and select only the target feature
    ts_series = ts_data.set_index('timestamp')[feature]
    
    # Handle missing values and infinity
    ts_series = ts_series.replace([np.inf, -np.inf], np.nan)
    
    # Convert to numeric type to ensure compatibility with ARIMA
    ts_series = pd.to_numeric(ts_series, errors='coerce')
    
    # Drop NaN values
    ts_series = ts_series.dropna()
    
    # If there's not enough data, raise an error
    if len(ts_series) < 3:
        return {
            'error': f"Not enough valid data points to train time series model for {feature}",
            'feature': feature
        }
    
    # Use simple default parameters
    p, d, q = 1, 1, 0
    
    try:
        # Train ARIMA model with simple parameters
        model = ARIMA(ts_series, order=(p, d, q))
        fitted_model = model.fit()
        
        # Forecast for the next 10 periods
        forecast_periods = 10
        forecast = fitted_model.forecast(steps=forecast_periods)
        
        # Calculate metrics
        training_metrics = {
            'mse': ((fitted_model.resid) ** 2).mean(),
            'mae': abs(fitted_model.resid).mean(),
            'residual_std': fitted_model.resid.std()
        }
        
        # Prepare result
        result = {
            'model': fitted_model,
            'forecast': forecast,
            'feature': feature,
            'last_timestamp': ts_series.index[-1],
            'order': (p, d, q),
            'training_metrics': training_metrics,
            'training_data': ts_series
        }
        
        return result
    except Exception as e:
        # If ARIMA fails, return error information
        return {
            'error': str(e),
            'feature': feature
        }

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
