import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def plot_feature_importance(feature_importance_df, top_n=10):
    """
    Plot feature importance from a model.
    
    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        DataFrame with feature importance scores
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get top N features
    if top_n is not None:
        data = feature_importance_df.head(top_n)
    else:
        data = feature_importance_df
    
    # Create horizontal bar plot
    sns.barplot(x='importance', y='feature', data=data)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, cm=None):
    """
    Plot confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    cm : array-like or None
        Pre-calculated confusion matrix (if available)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate confusion matrix if not provided
    if cm is None:
        cm = confusion_matrix(y_true, y_pred)
    elif isinstance(cm, list):
        # Convert list to numpy array if needed
        cm = np.array(cm)
    
    # Get a nice colormap with higher contrast
    cmap = plt.cm.Blues
    
    # Plot with seaborn for a nicer visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'],
                cbar=True, square=True, linewidths=0.5)
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    height, width = cm.shape
    for i in range(height):
        for j in range(width):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]/np.sum(cm):.1%})', 
                    horizontalalignment='center', color='black', fontsize=9)
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_metrics_over_time(data, time_series_model_dict):
    """
    Plot time series data with forecasts.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original data with timestamps
    time_series_model_dict : dict
        Dictionary with time series model and forecast
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot
    """
    # Create default figure for error cases
    fig = go.Figure()
    
    # Check if we have an error in the model dictionary
    if 'error' in time_series_model_dict:
        fig.add_annotation(
            text=f"Error: {time_series_model_dict['error']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Time Series Model Error")
        return fig

    # Extract model information with error handling
    try:
        model = time_series_model_dict.get('model')
        forecast = time_series_model_dict.get('forecast')
        feature = time_series_model_dict.get('feature')
        last_timestamp = time_series_model_dict.get('last_timestamp')
        
        # Properly check for missing components, avoiding Series truth value evaluation
        if (model is None or forecast is None or feature is None or last_timestamp is None or 
            (isinstance(forecast, pd.Series) and forecast.empty)):
            raise KeyError("Missing required time series model components")
            
        # Validate data format
        if 'timestamp' not in data.columns:
            raise ValueError("Data must contain a 'timestamp' column")
            
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' not found in data columns")
            
        # Prepare the data - handle potential data type issues
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Set index and extract feature
            historical_data = data.set_index('timestamp')[feature]
            
            # Create forecast timestamps (daily intervals from last timestamp)
            if isinstance(last_timestamp, str):
                last_timestamp = pd.to_datetime(last_timestamp)
                
            forecast_idx = pd.date_range(
                start=last_timestamp,
                periods=len(forecast) + 1,
                freq='D'
            )[1:]  # Skip first as it's the last historical timestamp
        except Exception as inner_e:
            raise ValueError(f"Data format error: {str(inner_e)}")
        
    except Exception as e:
        fig.add_annotation(
            text=f"Error processing time series data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Time Series Processing Error")
        return fig
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_idx,
        y=forecast.values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence intervals if available
    if hasattr(forecast, 'conf_int') and callable(getattr(forecast, 'conf_int')):
        try:
            conf_int = forecast.conf_int()
            if not conf_int.empty and conf_int.shape[1] >= 2:
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]
            else:
                # Skip confidence intervals if the data is not properly formatted
                lower = upper = None
        except Exception:
            # Skip confidence intervals on error
            lower = upper = None
        
        # Only add confidence interval traces if both lower and upper are valid
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=forecast_idx,
                y=upper,
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_idx,
                y=lower,
                mode='lines',
                name='Lower CI',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(width=0),
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title=f'{feature} - Time Series Forecast',
        xaxis_title='Timestamp',
        yaxis_title=feature,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99),
        height=500,
        width=800
    )
    
    return fig

def plot_anomaly_detection(X_test, predictions, n_components=2):
    """
    Visualize anomaly detection results using PCA for dimensionality reduction.
    
    Parameters:
    -----------
    X_test : pandas.DataFrame
        Test features
    predictions : array-like
        Predicted labels (1 for anomaly, 0 for normal)
    n_components : int
        Number of PCA components to use for visualization
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive scatter plot showing anomalies
    """
    # Apply PCA for visualization
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_test)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Prediction': ['Anomaly' if p == 1 else 'Normal' for p in predictions]
    })
    
    # Create Plotly figure
    fig = px.scatter(
        plot_df,
        x='PCA1',
        y='PCA2',
        color='Prediction',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        title='Anomaly Detection Visualization (PCA)',
        hover_data={'PCA1': True, 'PCA2': True, 'Prediction': True}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
        yaxis_title=f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
        height=600,
        width=800
    )
    
    return fig


def create_time_series_performance_matrix(metrics, model_name="ARIMA Time Series Model"):
    """
    Create a visual performance matrix for time series forecasting models.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of time series performance metrics
    model_name : str
        Name of the model to display in the matrix
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive performance matrix visualization
    """
    if metrics is None or not metrics.get('metrics_available', False):
        # Return an empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title="Time Series Performance Matrix",
            annotations=[
                dict(
                    text="No metrics available for time series model",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ],
            height=400,
            width=700
        )
        return fig
    
    # Extract metrics for display, handling potential missing metrics
    metric_names = [
        'RMSE', 'MAE', 'MAPE (%)', 'R²', 'Direction Accuracy (%)'
    ]
    
    metric_values = [
        round(metrics.get('rmse', float('nan')), 3),
        round(metrics.get('mae', float('nan')), 3),
        round(metrics.get('mape', float('nan')), 1),
        round(metrics.get('r2', float('nan')), 3),
        round(metrics.get('direction_accuracy', float('nan')) * 100, 1)
    ]
    
    # Create a color scale based on typical good/bad values for each metric
    # Lower is better for error metrics (RMSE, MAE, MAPE)
    # Higher is better for fit metrics (R², Direction Accuracy)
    
    # Normalize metric values to a 0-1 scale for coloring
    # Use inverted scale for error metrics (lower is better)
    # These are placeholder values for normalization and should be adjusted
    # based on domain knowledge and data characteristics
    max_vals = [10, 10, 50, 1, 100]  # Maximum expected values
    min_vals = [0, 0, 0, -1, 0]      # Minimum expected values
    
    # Calculate normalized values (0-1 scale)
    normalized_values = []
    for i, val in enumerate(metric_values):
        if np.isnan(val):
            normalized_values.append(0.5)  # Middle value for missing metrics
            continue
            
        # For error metrics (first 3), lower is better, so invert
        if i < 3:  # RMSE, MAE, MAPE
            norm_val = 1 - max(0, min(1, (val - min_vals[i]) / (max_vals[i] - min_vals[i])))
        else:  # R², Direction Accuracy
            norm_val = max(0, min(1, (val - min_vals[i]) / (max_vals[i] - min_vals[i])))
            
        normalized_values.append(norm_val)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[normalized_values],
        x=metric_names,
        y=[model_name],
        colorscale='RdYlGn',  # Red (bad) to Green (good)
        showscale=False,
        text=[[str(val) for val in metric_values]],
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    # Add any warnings as annotations
    annotations = []
    for key, value in metrics.items():
        if key.startswith('warning'):
            annotations.append(
                dict(
                    text=value,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    font=dict(size=10, color="red")
                )
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Time Series Forecasting Performance Matrix',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=250,
        width=700,
        annotations=annotations,
        margin=dict(l=50, r=50, t=80, b=80)
    )
    
    return fig

def create_classification_performance_matrix(metrics, model_name="Classification Model"):
    """
    Create a visual performance matrix for classification models.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of classification performance metrics
    model_name : str
        Name of the model to display in the matrix
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive performance matrix visualization
    """
    if metrics is None or not metrics:
        # Return an empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title="Classification Performance Matrix",
            annotations=[
                dict(
                    text="No metrics available for classification model",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ],
            height=250,
            width=700
        )
        return fig
    
    # Extract metrics for display, handling potential missing metrics
    metric_names = [
        'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'
    ]
    
    metric_values = [
        round(float(metrics.get('accuracy', 0)), 3),
        round(float(metrics.get('precision', 0)), 3),
        round(float(metrics.get('recall', 0)), 3),
        round(float(metrics.get('f1', 0)), 3),
        round(float(metrics.get('auc', 0)), 3) if metrics.get('auc') is not None else None
    ]
    
    # Filter out None values if AUC is not available
    valid_metrics = [(name, val) for name, val in zip(metric_names, metric_values) if val is not None]
    if valid_metrics:
        metric_names, metric_values = zip(*valid_metrics)
    
    # Create a color scale based on typical good/bad values
    # For classification metrics, typically higher is better
    # These values can be tuned based on domain knowledge
    
    # Normalize metric values to a 0-1 scale for coloring
    # All metrics here are on a 0-1 scale already, but we can still customize
    # the color thresholds
    normalized_values = []
    for val in metric_values:
        if val is None:
            normalized_values.append(0.5)  # Middle value for missing metrics
        else:
            # For classification metrics (all), higher is better
            # Below 0.5 is considered poor, 0.7-0.8 is good, 0.9+ is excellent
            normalized_values.append(val)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[normalized_values],
        x=metric_names,
        y=[model_name],
        colorscale='RdYlGn',  # Red (bad) to Green (good)
        showscale=False,
        text=[[str(val) for val in metric_values]],
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Classification Model Performance Matrix',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=200,
        width=700,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig