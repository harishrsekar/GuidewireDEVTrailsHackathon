import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(data):
    """
    Preprocess the Kubernetes metrics data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw Kubernetes metrics data
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    dict
        Preprocessing metadata
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Metadata to store preprocessing info
    metadata = {}
    
    # Handle missing values
    missing_counts = df.isnull().sum()
    metadata['missing_values'] = missing_counts.to_dict()
    
    # For numeric columns, fill NaNs with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill NaNs with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        metadata['timestamp_converted'] = True
    
    # Encode categorical variables if any (except 'node' which we'll keep as-is for reference)
    categorical_cols = [col for col in categorical_cols if col != 'node' and col != 'timestamp']
    for col in categorical_cols:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    # Drop any non-numeric columns that aren't needed for modeling
    # except 'timestamp' which we keep for time-series analysis
    cols_to_drop = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) 
                  and col != 'timestamp' and col != 'failure']
    
    if cols_to_drop:
        metadata['dropped_columns'] = cols_to_drop
        df = df.drop(columns=cols_to_drop)
    
    # Ensure 'failure' is binary (0 or 1)
    if 'failure' in df.columns:
        df['failure'] = df['failure'].astype(int)
    
    return df, metadata

def apply_feature_engineering(data, window_sizes=[5, 10], apply_pca=False, n_components=None):
    """
    Apply feature engineering to enhance the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Preprocessed data
    window_sizes : list
        List of window sizes for rolling statistics
    apply_pca : bool
        Whether to apply PCA
    n_components : int
        Number of PCA components
    
    Returns:
    --------
    pandas.DataFrame
        Data with engineered features
    """
    df = data.copy()
    
    # Keep track of the original columns
    original_feature_cols = [col for col in df.columns if col != 'failure' and col != 'timestamp']
    
    # If timestamp exists, calculate temporal features
    if 'timestamp' in df.columns:
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Extract temporal features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Calculate rolling statistics for each window size
        for window in window_sizes:
            for col in original_feature_cols:
                # Skip columns that are likely categorical
                if df[col].nunique() < 5 or '_condition_' in col:
                    continue
                
                # Rolling mean
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                
                # Rolling min
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                
                # Rolling max
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
                # Rate of change
                df[f'{col}_roc_{window}'] = df[col].pct_change(periods=window).fillna(0)
    
    # Create interaction features for important metrics
    if 'cpu_usage_percent' in df.columns and 'memory_usage_percent' in df.columns:
        df['cpu_memory_product'] = df['cpu_usage_percent'] * df['memory_usage_percent']
    
    if 'pod_restart_count' in df.columns and 'pod_pending_count' in df.columns:
        df['pod_issues_sum'] = df['pod_restart_count'] + df['pod_pending_count']
    
    # Create threshold-based features
    if 'cpu_usage_percent' in df.columns:
        df['high_cpu'] = (df['cpu_usage_percent'] > 80).astype(int)
    
    if 'memory_usage_percent' in df.columns:
        df['high_memory'] = (df['memory_usage_percent'] > 80).astype(int)
    
    if 'disk_usage_percent' in df.columns:
        df['high_disk'] = (df['disk_usage_percent'] > 80).astype(int)
    
    # Apply PCA if requested
    if apply_pca and n_components is not None:
        # Keep only numeric features except 'failure' and 'timestamp'
        numeric_cols = [col for col in df.columns 
                      if pd.api.types.is_numeric_dtype(df[col]) 
                      and col != 'failure' 
                      and col != 'timestamp']
        
        # Ensure n_components isn't larger than number of features
        n_components = min(n_components, len(numeric_cols))
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        # Clean numeric data before PCA
    numeric_data = df[numeric_cols].copy()
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    # Apply standard scaling before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Now apply PCA on cleaned and scaled data
    pca_result = pca.fit_transform(scaled_data)
        
        # Add PCA components to dataframe
        for i in range(n_components):
            df[f'pca_component_{i+1}'] = pca_result[:, i]
    
    # Drop timestamp column as it's not needed for modeling
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    return df

def handle_class_imbalance(X, y, method='SMOTE'):
    """
    Handle class imbalance in the dataset.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    method : str
        Method to handle imbalance ('SMOTE', 'RandomUnderSampler', 'ADASYN')
    
    Returns:
    --------
    pandas.DataFrame
        Balanced feature matrix
    pandas.Series
        Balanced target variable
    """
    if method == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif method == 'RandomUnderSampler':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'ADASYN':
        sampler = ADASYN(random_state=42)
    else:
        return X, y
    
    # Perform resampling
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled

def apply_scaling(X_train, X_test, method='StandardScaler'):
    """
    Apply scaling to the features.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    X_test : pandas.DataFrame
        Testing feature matrix
    method : str
        Scaling method ('StandardScaler', 'RobustScaler', 'MinMaxScaler')
    
    Returns:
    --------
    pandas.DataFrame
        Scaled training features
    pandas.DataFrame
        Scaled testing features
    scaler
        Fitted scaler object
    """
    if method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'RobustScaler':
        scaler = RobustScaler()
    elif method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        # Default to StandardScaler
        scaler = StandardScaler()
    
    # Fit on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Transform test data
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler
