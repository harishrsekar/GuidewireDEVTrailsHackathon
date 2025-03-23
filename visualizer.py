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
        
        if not all([model, forecast, feature, last_timestamp]):
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
    if hasattr(forecast, 'conf_int'):
        conf_int = forecast.conf_int()
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        
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
