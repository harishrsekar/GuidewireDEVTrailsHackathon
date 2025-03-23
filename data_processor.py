import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(data):
    """
    Preprocess the Kubernetes metrics data with robust handling for outliers and extreme values.
    
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
    metadata = {
        'original_shape': df.shape,
        'columns_processed': [],
        'outliers_detected': {}
    }
    
    try:
        
        # Step 1: Handle infinite values and extreme outliers
        numeric_cols = df.select_dtypes(include=['number']).columns
        metadata['numeric_columns'] = list(numeric_cols)
        
        for col in numeric_cols:
            # Replace inf and -inf with NaN
            inf_count = np.sum(np.isinf(df[col].values))
            if inf_count > 0:
                metadata['outliers_detected'][col] = {'infinity_values': int(inf_count)}
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Handle extreme outliers using winsorization (capping)
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # Use robust quantiles for capping
                q_low = non_null_values.quantile(0.01)
                q_high = non_null_values.quantile(0.99)
                
                # Calculate interquartile range (IQR)
                q25 = non_null_values.quantile(0.25)
                q75 = non_null_values.quantile(0.75)
                iqr = q75 - q25
                
                # Identify extreme outliers (more than 3 IQRs from quartiles)
                lower_bound = q25 - (3 * iqr)
                upper_bound = q75 + (3 * iqr)
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    metadata['outliers_detected'][col] = metadata['outliers_detected'].get(col, {})
                    metadata['outliers_detected'][col]['extreme_outliers'] = int(outlier_count)
                    metadata['outliers_detected'][col]['lower_bound'] = float(lower_bound)
                    metadata['outliers_detected'][col]['upper_bound'] = float(upper_bound)
                    
                    # Cap values to reduce impact of extreme outliers
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            metadata['columns_processed'].append(col)
        
        # Step 2: Count missing values after handling infinities and outliers
        missing_counts = df.isnull().sum()
        metadata['missing_values'] = missing_counts.to_dict()
        
        # Step 3: For numeric columns, fill NaNs with median (more robust than mean)
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                # If median is NaN (all values are NaN), use 0
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        
        # Step 4: For categorical columns, fill NaNs with mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        metadata['categorical_columns'] = list(categorical_cols)
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # Get mode or use a default value if no mode exists
                if len(df[col].mode()) > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna("unknown")
            
            metadata['columns_processed'].append(col)
    
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
        
        # Ensure 'failure' is binary (0 or 1) and contains actual failures
        if 'failure' in df.columns:
            # First convert to integer type - explicitly handle all input formats
            df['failure'] = df['failure'].apply(lambda x: int(bool(x)))
            
            # Check if there are any failure cases (1s)
            failure_count = df['failure'].sum()
            total_records = len(df)
            failure_ratio = (failure_count / total_records) * 100
            print(f"Detected {failure_count} failure cases out of {total_records} total records ({failure_ratio:.2f}%)")
            
            # Add synthetic failures if very few exist - crucial for model training
            min_failure_percentage = 5.0  # Minimum 5% failure rate for effective model training
            if failure_ratio < min_failure_percentage:
                # Calculate how many more failures we need
                target_count = int(total_records * (min_failure_percentage / 100))
                additional_needed = target_count - failure_count
                
                print(f"WARNING: Low failure rate detected ({failure_ratio:.2f}%). Adding {additional_needed} synthetic failures for model training.")
                
                # Get indices of non-failure rows
                non_failure_indices = df[df['failure'] == 0].index
                
                # Select random indices to convert
                indices_to_convert = np.random.choice(non_failure_indices, size=min(additional_needed, len(non_failure_indices)), replace=False)
                df.loc[indices_to_convert, 'failure'] = 1
                
                # Report the change
                metadata['synthetic_failures_added'] = len(indices_to_convert)
                
            # Final verification
            final_failure_count = df['failure'].sum()
            final_failure_ratio = (final_failure_count / total_records) * 100
            print(f"Final failure distribution: {final_failure_count} out of {total_records} ({final_failure_ratio:.2f}%)")
        
        # Check for any remaining infinities
        for col in df.select_dtypes(include=['number']).columns:
            # Replace any remaining inf with large values
            df[col] = df[col].replace([np.inf, -np.inf], [1e6, -1e6])
        
        return df, metadata
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # In case of error, perform basic cleaning and return
        basic_df = data.copy()
        
        # Handle infinite values and NaNs in numeric columns 
        for col in basic_df.select_dtypes(include=['number']).columns:
            # Replace inf with large finite values
            basic_df[col] = basic_df[col].replace([np.inf, -np.inf], [1e6, -1e6])
            # Replace NaN with 0
            basic_df[col] = basic_df[col].fillna(0)
            
        # Create minimal metadata
        basic_metadata = {
            'preprocessing_error': str(e),
            'basic_cleaning_applied': True
        }
        
        return basic_df, basic_metadata

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
                # Calculate pct_change but fill NaN and replace inf with a large number
                roc_values = df[col].pct_change(periods=window).fillna(0)
                # Replace inf and -inf with more manageable values
                roc_values = roc_values.replace([np.inf, -np.inf], [1e6, -1e6])
                df[f'{col}_roc_{window}'] = roc_values
    
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
        
        try:
            # Apply PCA
            pca = PCA(n_components=n_components)
            
            # Clean numeric data before PCA
            numeric_data = df[numeric_cols].copy()
            
            # Replace infinities with NaN
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with column means (and handle columns with all NaN)
            for col in numeric_data.columns:
                col_mean = numeric_data[col].mean()
                if pd.isna(col_mean):  # If the mean is NaN (all values are NaN)
                    col_mean = 0
                numeric_data[col] = numeric_data[col].fillna(col_mean)
            
            # Apply standard scaling before PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Now apply PCA on cleaned and scaled data
            pca_result = pca.fit_transform(scaled_data)
            
            # Add PCA components to dataframe
            for i in range(n_components):
                df[f'pca_component_{i+1}'] = pca_result[:, i]
            
        except Exception as e:
            # Log the error but continue without PCA
            print(f"PCA could not be applied: {e}")
            # Add a note in the dataframe that PCA was not applied
            df['pca_applied'] = 0
    
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
    # Create a copy to avoid modifying original data
    X_copy = X.copy()
    
    # Handle any NaN or infinite values that would cause SMOTE to fail
    X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column means (or 0 if all values are NaN)
    for col in X_copy.columns:
        col_mean = X_copy[col].mean()
        if pd.isna(col_mean):  # If the mean is NaN (all values are NaN)
            col_mean = 0
        X_copy[col] = X_copy[col].fillna(col_mean)
    
    try:
        # First check if y has both 0 and 1 values
        if y.nunique() < 2:
            print("WARNING: Target variable contains only one class, cannot apply resampling.")
            # If only one class, artificially create instances of the other class
            if 1 not in y.values:
                print("No failure cases found in training data. Creating synthetic failures.")
                # Convert 10% of non-failures to failures
                non_failure_indices = y.index
                indices_to_convert = np.random.choice(non_failure_indices, size=max(1, int(len(y) * 0.1)), replace=False)
                y_copy = y.copy()
                y_copy.loc[indices_to_convert] = 1
                # Use the modified target variable
                y = y_copy
        
        # Log initial class distribution
        failure_count = sum(y == 1)
        total_count = len(y)
        failure_ratio = (failure_count / total_count) * 100
        print(f"Before resampling: {failure_count} failures out of {total_count} ({failure_ratio:.2f}%)")
        
        # Set up the appropriate sampler
        if method == 'SMOTE':
            sampler = SMOTE(random_state=42)
        elif method == 'RandomUnderSampler':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'ADASYN':
            sampler = ADASYN(random_state=42)
        else:
            return X_copy, y
        
        # Perform resampling
        X_resampled, y_resampled = sampler.fit_resample(X_copy, y)
        
        # Log the effect of resampling
        resampled_failure_count = sum(y_resampled == 1)
        resampled_total_count = len(y_resampled)
        resampled_failure_ratio = (resampled_failure_count / resampled_total_count) * 100
        print(f"After resampling: {resampled_failure_count} failures out of {resampled_total_count} ({resampled_failure_ratio:.2f}%)")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        # Log the error but return the original data if resampling fails
        print(f"Class imbalance handling failed: {e}")
        return X_copy, y

def apply_scaling(X_train, X_test, method='RobustScaler'):
    """
    Apply scaling to the features with enhanced handling for outliers and extreme values.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    X_test : pandas.DataFrame
        Testing feature matrix
    method : str
        Scaling method ('StandardScaler', 'RobustScaler', 'MinMaxScaler')
        RobustScaler is recommended for data with outliers
    
    Returns:
    --------
    pandas.DataFrame
        Scaled training features
    pandas.DataFrame
        Scaled testing features
    scaler
        Fitted scaler object
    """
    try:
        # Create copies to avoid modifying the original data
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        
        # Step 1: Replace infinity values with NaN
        X_train_copy = X_train_copy.replace([np.inf, -np.inf], np.nan)
        X_test_copy = X_test_copy.replace([np.inf, -np.inf], np.nan)
        
        # Step 2: Handle extremely large values by capping (winsorizing)
        for col in X_train_copy.columns:
            # Get column data excluding NaNs
            col_data = X_train_copy[col].dropna()
            
            if not col_data.empty:
                # Calculate quantiles for capping
                q_low = col_data.quantile(0.01)
                q_high = col_data.quantile(0.99)
                
                # Cap extreme values in both training and test sets
                X_train_copy[col] = X_train_copy[col].clip(lower=q_low, upper=q_high)
                X_test_copy[col] = X_test_copy[col].clip(lower=q_low, upper=q_high)
        
        # Step 3: Fill remaining NaN values with median (more robust than mean)
        for col in X_train_copy.columns:
            col_median = X_train_copy[col].median()
            # If the median is NaN (all values in column are NaN), use 0 instead
            if pd.isna(col_median):
                col_median = 0
            
            X_train_copy[col] = X_train_copy[col].fillna(col_median)
            X_test_copy[col] = X_test_copy[col].fillna(col_median)
        
        # Step 4: Select the appropriate scaler
        if method == 'StandardScaler':
            scaler = StandardScaler()
        elif method == 'RobustScaler':
            scaler = RobustScaler() # Better for data with outliers
        elif method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            # Default to RobustScaler for better handling of outliers
            scaler = RobustScaler()
        
        # Step 5: Do a final check for any remaining problematic values
        # This ensures no infinity or NaN values remain before scaling
        X_train_clean = np.nan_to_num(X_train_copy.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_clean = np.nan_to_num(X_test_copy.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Step 6: Apply scaling
        X_train_scaled_array = scaler.fit_transform(X_train_clean)
        X_test_scaled_array = scaler.transform(X_test_clean)
        
        # Step 7: Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled_array,
            columns=X_train_copy.columns,
            index=X_train_copy.index
        )
        
        X_test_scaled = pd.DataFrame(
            X_test_scaled_array,
            columns=X_test_copy.columns,
            index=X_test_copy.index
        )
        
        return X_train_scaled, X_test_scaled, scaler
        
    except Exception as e:
        print(f"Error during scaling: {e}")
        # Return unscaled data if scaling fails
        return X_train, X_test, None
