import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc,
                           classification_report, mean_squared_error, mean_absolute_error, r2_score)
from math import sqrt

def evaluate_model(model, X_test, y_test, model_type='random_forest'):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : trained model
        The trained model to evaluate
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test targets
    model_type : str
        Type of model ('random_forest' or 'isolation_forest')
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    results = {}
    
    if model_type == 'random_forest':
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, zero_division=0)
        results['recall'] = recall_score(y_test, y_pred, zero_division=0)
        results['f1'] = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        results['auc'] = auc(fpr, tpr)
        results['fpr'] = fpr
        results['tpr'] = tpr
        
        # Store predictions and probabilities
        results['predictions'] = y_pred
        results['probabilities'] = y_pred_proba
        
        # Generate detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        
        # Calculate and store confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
        
    elif model_type == 'isolation_forest':
        # For Isolation Forest, -1 is anomaly, 1 is normal
        # Convert to 1 for anomaly (failure), 0 for normal to match our target
        y_pred_raw = model.predict(X_test)
        y_pred = np.where(y_pred_raw == -1, 1, 0)
        
        # Calculate anomaly scores
        scores = model.score_samples(X_test)
        
        # Calculate metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, zero_division=0)
        results['recall'] = recall_score(y_test, y_pred, zero_division=0)
        results['f1'] = f1_score(y_test, y_pred, zero_division=0)
        
        # Store predictions and scores
        results['predictions'] = y_pred
        results['anomaly_scores'] = scores
        
        # Generate detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        
        # Calculate and store confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
    
    return results

def get_feature_importance(model, feature_names, top_n=None):
    """
    Extract feature importance from the model.
    
    Parameters:
    -----------
    model : trained model
        The trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int or None
        Number of top features to return, or None for all
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature names and importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Create DataFrame of feature importances
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    # Sort by importance
    importances = importances.sort_values('importance', ascending=False)
    
    # Get top N if specified
    if top_n is not None:
        importances = importances.head(top_n)
    
    return importances


def evaluate_time_series_model(actual_values, predicted_values, model_info=None):
    """
    Evaluate a time series forecasting model.
    
    Parameters:
    -----------
    actual_values : pandas.Series or array-like
        The actual observed values
    predicted_values : pandas.Series or array-like
        The forecasted values from the model
    model_info : dict, optional
        Additional model information like AIC, BIC
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics for time series forecasting
    """
    results = {}

    # Make sure we have valid arrays to compare
    if actual_values is None or predicted_values is None:
        return {
            'error': 'Missing actual or predicted values',
            'metrics_available': False
        }
    
    # Convert to numpy arrays to ensure consistent handling
    actual = np.array(actual_values) if not isinstance(actual_values, np.ndarray) else actual_values
    predicted = np.array(predicted_values) if not isinstance(predicted_values, np.ndarray) else predicted_values
    
    # Check if arrays are empty or have different lengths
    if len(actual) == 0 or len(predicted) == 0:
        return {
            'error': 'Empty actual or predicted values',
            'metrics_available': False
        }
    
    # If lengths are different, truncate to the shorter one
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        results['warning'] = f'Different lengths in actual ({len(actual_values)}) and predicted ({len(predicted_values)}) values. Truncated to {min_len}.'
    
    # Calculate standard regression metrics
    results['mse'] = mean_squared_error(actual, predicted)
    results['rmse'] = sqrt(results['mse'])
    results['mae'] = mean_absolute_error(actual, predicted)
    
    # Add R-squared if there is variance in the actual values
    if np.var(actual) > 0:
        results['r2'] = r2_score(actual, predicted)
    else:
        results['r2'] = np.nan
        results['warning_r2'] = 'No variance in actual values, R-squared is undefined'
    
    # Calculate mean absolute percentage error (MAPE), avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((actual - predicted) / actual) * 100
        # Replace infinities with NaN
        mape_values = np.where(np.isinf(mape_values), np.nan, mape_values)
        # Calculate mean excluding NaN values
        if not np.all(np.isnan(mape_values)):
            results['mape'] = np.nanmean(mape_values)
        else:
            results['mape'] = np.nan
            results['warning_mape'] = 'MAPE could not be calculated due to zero values in actual data'
    
    # Add model information if available
    if model_info is not None:
        if 'aic' in model_info:
            results['aic'] = model_info['aic']
        if 'bic' in model_info:
            results['bic'] = model_info['bic']
    
    # Add forecast accuracy
    # Direction accuracy: percentage of times the forecast correctly predicts the direction of change
    if len(actual) > 1 and len(predicted) > 1:
        actual_diff = np.diff(actual)
        pred_diff = np.diff(predicted)
        
        # Count correct directions (both positive or both negative)
        correct_dirs = np.sum((actual_diff > 0) & (pred_diff > 0)) + np.sum((actual_diff < 0) & (pred_diff < 0))
        results['direction_accuracy'] = float(correct_dirs) / float(len(actual_diff)) if len(actual_diff) > 0 else np.nan
    
    # Flag that metrics are available
    results['metrics_available'] = True
    
    return results
