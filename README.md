# Kubernetes Failure Prediction Application

A Streamlit-based machine learning application that predicts failures in Kubernetes clusters through analysis of historical metrics data. This tool helps DevOps teams proactively identify potential node crashes, resource exhaustion, and other common failure scenarios before they impact production systems.

## Application Overview

The application provides an end-to-end pipeline for Kubernetes cluster health monitoring and prediction:

1. **Data Preparation** - Generate synthetic data or upload your own metrics for analysis
2. **Model Training** - Train various ML models including Random Forest, Isolation Forest, and ARIMA
3. **Model Evaluation** - Evaluate model performance with detailed metrics
4. **Prediction** - Run predictions on new data to identify potential failures

## Video Demo

https://github.com/user-attachments/assets/19a4fa7e-c59b-4c66-b15d-f5a9e7ac144c

## Features

- **Interactive Web Interface**: Built with Streamlit for intuitive user interaction
- **Multiple Machine Learning Models**:
  - Random Forest for classification-based prediction
  - Isolation Forest for anomaly detection
  - ARIMA for time series forecasting
- **Comprehensive Data Processing**:
  - Automated preprocessing with robust handling for outliers
  - Advanced feature engineering with rolling statistics
  - Class imbalance handling with SMOTE and other techniques
- **Detailed Visualization**:
  - Feature importance plots
  - Confusion matrices
  - Time series forecasts
  - Performance metric dashboards

## Model Architecture & Technical Details

### Random Forest Classifier

The Random Forest model is used for classification-based prediction of Kubernetes failures. It works by:

1. Creating multiple decision trees during training
2. Using majority voting to make final predictions
3. Providing feature importance to identify critical metrics affecting cluster health

**Key Parameters:**
- Number of estimators (trees): 50-300 (configurable)
- Maximum depth: 5-30 (configurable)
- Class weights: Automatically adjusted for imbalanced datasets

<img src="https://github.com/user-attachments/assets/74dc6403-df66-460f-8368-11bc8a26d12f" width="500">

### Isolation Forest

The Isolation Forest model is used for anomaly detection in Kubernetes metrics. It:

1. Isolates observations by randomly selecting features
2. Creates shorter paths for anomalies compared to normal observations
3. Identifies potential failures without relying on historical failure labels

**Key Parameters:**
- Contamination: Automatically calculated based on expected failure rate
- Number of estimators: 50-300 (configurable)

<img src="https://github.com/user-attachments/assets/f1e82da6-cf64-4025-b1d9-801908fbce9f" width="500">

### ARIMA Time Series Model

The ARIMA (AutoRegressive Integrated Moving Average) model is used for forecasting future metrics values and detecting trends. It:

1. Captures temporal patterns in metrics data
2. Forecasts future values based on historical patterns
3. Helps identify potential deviations that could indicate upcoming failures

<img src="https://github.com/user-attachments/assets/6b2086c8-4ce1-4655-9383-3a3d6f6eef06" width="700">

## Performance Metrics

The application provides comprehensive performance metrics for each model:

### Classification Metrics (Random Forest)
- Accuracy: Overall prediction accuracy
- Precision: Ability to correctly identify failures without false alarms
- Recall: Ability to detect all actual failures
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Area under the Receiver Operating Characteristic curve

<img src="https://github.com/user-attachments/assets/341e8b9a-c69d-4b62-9d40-a4ffe849565a" width="800">

### Anomaly Detection Metrics (Isolation Forest)
- Anomaly Score Distribution
- Precision and Recall at various thresholds
- F1 Score optimization for threshold selection

<img src="https://github.com/user-attachments/assets/67d225ed-a23e-41b6-a10c-bdba976b666c" width="800">

### Time Series Metrics (ARIMA)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Direction Accuracy

<img src="https://github.com/user-attachments/assets/b8811bae-6163-46e7-8ea8-252dfe03dfbb" width="800">

## Data Processing Pipeline

The application implements a robust data processing pipeline:

1. **Preprocessing**:
   - Handling missing values
   - Outlier detection and treatment
   - Data normalization
   
2. **Feature Engineering**:
   - Temporal feature creation (rolling statistics)
   - Optional dimensionality reduction with PCA
   - Automated feature selection
   
3. **Class Imbalance Handling**:
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Random Under-sampling
   - ADASYN (Adaptive Synthetic Sampling)
   
4. **Scaling**:
   - Standard Scaling
   - Robust Scaling (better for data with outliers)
   - Min-Max Scaling

## Prerequisites

To run the application, you need:

### Software Requirements
- Python 3.9+
- pip (Python package installer)

### Python Packages
- streamlit: For the web interface
- pandas, numpy: For data manipulation
- scikit-learn: For machine learning models
- imbalanced-learn: For handling class imbalance
- matplotlib, plotly, seaborn: For visualization
- statsmodels: For time series modeling
- joblib: For model persistence

These dependencies can be installed using:
```bash
pip install -r requirements.txt
```

### Hardware Recommendations
- CPU: 2+ cores recommended for model training
- RAM: 4GB+ (8GB+ recommended for larger datasets)
- Storage: 1GB for application and dependencies

## Running the Application

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kubernetes-failure-prediction.git
cd kubernetes-failure-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
streamlit run app.py
```

4. Access the application in your web browser at `http://localhost:5000`

## Usage Guide

### Step 1: Data Preparation
- Select "Generate Sample Data" or upload your own dataset
- Adjust failure rate and sample size if generating data
- Preprocess the data with the available options

### Step 2: Model Training
- Train at least one of the available models
- Adjust model parameters as needed
- Review feature importance plots for the Random Forest model

### Step 3: Model Evaluation
- Evaluate model performance using the various metrics
- Compare different models to select the best one
- Analyze confusion matrices and performance visualizations

### Step 4: Prediction
- Upload new data or use existing test data
- Run predictions to identify potential failures
- Review prediction results and detailed explanations

## Input Parameters and Attributes

The application uses the following Kubernetes metrics as input parameters for prediction models:

### Node-level Metrics
1. **CPU Usage** (`cpu_usage_percent`) - Percentage of CPU utilization on the node
2. **Memory Usage** (`memory_usage_percent`) - Percentage of memory usage on the node
3. **Disk Usage** (`disk_usage_percent`) - Percentage of disk space utilized
4. **Network Activity**:
   - `network_receive_bytes` - Bytes received over the network
   - `network_transmit_bytes` - Bytes transmitted over the network

### Pod-related Metrics
1. **Pod Count** (`pod_count`) - Number of pods running on the node
2. **Pod Restart Count** (`pod_restart_count`) - Number of pod restarts
3. **Pod Pending Count** (`pod_pending_count`) - Number of pods pending scheduling

### Node Condition Indicators
1. **Node Ready Status** (`node_condition_ready`) - 1 if ready, 0 if not ready
2. **Memory Pressure** (`node_condition_memory_pressure`) - 1 if true, 0 if false
3. **Disk Pressure** (`node_condition_disk_pressure`) - 1 if true, 0 if false
4. **PID Pressure** (`node_condition_pid_pressure`) - 1 if true, 0 if false
5. **Network Unavailable** (`node_condition_network_unavailable`) - 1 if true, 0 if false

### Resource Quota Metrics
1. **CPU Request Percentage** (`cpu_request_percentage`) - Percentage of CPU resources requested
2. **Memory Request Percentage** (`memory_request_percentage`) - Percentage of memory resources requested

### Failure Patterns Detected

The models are trained to identify several types of failure patterns:

1. **CPU Exhaustion** - Characterized by:
   - High CPU usage (>85%)
   - Increased pod restart counts

2. **Memory Exhaustion** - Characterized by:
   - High memory usage (>90%)
   - Memory pressure condition

3. **Disk Pressure** - Characterized by:
   - High disk usage (>85%)
   - Disk pressure condition

4. **Network Issues** - Characterized by:
   - Abnormally low network throughput
   - Network unavailable condition

5. **Node Not Ready** - Characterized by:
   - Node condition ready = 0
   - Increased pending pod count

These input parameters are processed through feature engineering steps that create additional derived metrics, such as:
- Rolling averages over different time windows
- Rate of change calculations
- Statistical features (variance, skewness, etc.)
- Anomaly scores


## Extending the Application

The application can be extended in several ways:

1. **Additional Models**: Implement other ML algorithms like XGBoost, Neural Networks, etc.
2. **Real-time Integration**: Connect with Kubernetes API for real-time metrics
3. **Alert System**: Add notification capabilities for predicted failures
4. **Custom Metrics**: Add support for additional Kubernetes metrics
5. **Model Versioning**: Implement model versioning and A/B testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
