import pickle
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from datetime import datetime

def save_model(model, filename):
    """
    Save a trained model to a file.
    
    Parameters:
    -----------
    model : trained model
        The model to save
    filename : str
        Path where the model will be saved
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(filename):
    """
    Load a model from a file.
    
    Parameters:
    -----------
    filename : str
        Path to the model file
    
    Returns:
    --------
    object
        The loaded model
    """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def export_results(dataframe, filename=None):
    """
    Export DataFrame to CSV.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Data to export
    filename : str or None
        Filename for the export, if None a default will be used
    
    Returns:
    --------
    str
        Path to the exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"k8s_prediction_results_{timestamp}.csv"
    
    try:
        dataframe.to_csv(filename, index=False)
        return filename
    except Exception as e:
        print(f"Error exporting data: {e}")
        return None

def download_dataframe(dataframe, label="Download data as CSV"):
    """
    Create a download button for a DataFrame in Streamlit.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Data to offer for download
    label : str
        Label to show on the download button
    
    Returns:
    --------
    None
    """
    csv = dataframe.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"k8s_data_{timestamp}.csv"
    
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

def calculate_metrics_summary(predictions, actual=None):
    """
    Calculate summary statistics from predictions.
    
    Parameters:
    -----------
    predictions : array-like
        Array of predictions (1 for failure, 0 for normal)
    actual : array-like or None
        Array of actual values, if available
    
    Returns:
    --------
    dict
        Dictionary of summary metrics
    """
    summary = {}
    
    # Total predictions
    summary['total_count'] = len(predictions)
    
    # Failure count and percentage
    failure_count = np.sum(predictions)
    summary['failure_count'] = failure_count
    summary['failure_percentage'] = (failure_count / len(predictions)) * 100
    
    # Calculate accuracy metrics if actual values are provided
    if actual is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        summary['accuracy'] = accuracy_score(actual, predictions)
        summary['precision'] = precision_score(actual, predictions, zero_division=0)
        summary['recall'] = recall_score(actual, predictions, zero_division=0)
        summary['f1'] = f1_score(actual, predictions, zero_division=0)
    
    return summary

def format_prediction_result(model_name, prediction_data, error=None, performance_metrics=None):
    """
    Create a standardized prediction result object for consistent UI display.
    
    Parameters:
    -----------
    model_name : str
        Name of the model making the prediction
    prediction_data : dict or any
        Raw prediction data from the model
    error : str or None
        Error message if prediction failed
    performance_metrics : dict or None
        Model performance metrics from evaluation
    
    Returns:
    --------
    dict
        Standardized prediction result object
    """
    result = {
        'model': model_name.replace('_', ' ').title(),
        'timestamp': datetime.now().isoformat(),
        'status': 'error' if error else 'success'
    }
    
    if error:
        result['error'] = str(error)
        result['error_message'] = str(error)  # Added for compatibility with UI
        result['prediction'] = None
    else:
        result.update(prediction_data)
        
        # Include performance metrics if available
        if performance_metrics:
            result['performance_metrics'] = performance_metrics
    
    return result

def handle_prediction_errors(func):
    """
    Decorator for handling prediction function errors
    
    Parameters:
    -----------
    func : function
        The prediction function to wrap
    
    Returns:
    --------
    function
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            error_details = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'status': 'error'
            }
            st.error(f"Prediction error: {str(e)}")
            return error_details
    
    return wrapper
