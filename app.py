import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import time
import io

from data_generator import generate_kubernetes_data
from data_processor import (preprocess_data, apply_feature_engineering, 
                            handle_class_imbalance, apply_scaling)
from model_trainer import train_random_forest, train_isolation_forest, train_time_series_model
from model_evaluator import evaluate_model, get_feature_importance
from visualizer import (plot_feature_importance, plot_confusion_matrix, 
                        plot_metrics_over_time, plot_anomaly_detection)
from utils import save_model, load_model

# Set page configuration
st.set_page_config(page_title="Kubernetes Failure Prediction", 
                   page_icon="🔮", 
                   layout="wide")

# App title and description
st.title("🔮 Kubernetes Failure Prediction")
st.markdown("""
This application uses machine learning to predict failures in Kubernetes clusters based on historical metrics.
The models can help identify potential node crashes, resource exhaustion, and other common failure scenarios.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Preparation", "Model Training", "Model Evaluation", "Prediction"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'feature_engineered_data' not in st.session_state:
    st.session_state.feature_engineered_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Data Preparation Page
if page == "Data Preparation":
    st.header("1. Data Preparation")
    
    data_option = st.radio("Select data source:", ["Generate Sample Data", "Upload Your Own"])
    
    if data_option == "Generate Sample Data":
        st.subheader("Generate Kubernetes Metrics Sample Data")
        
        num_samples = st.slider("Number of samples to generate:", 1000, 10000, 5000)
        failure_rate = st.slider("Failure rate (%):", 1, 50, 10)
        time_steps = st.slider("Time periods to simulate:", 10, 100, 30)
        
        if st.button("Generate Data"):
            with st.spinner("Generating Kubernetes metrics data..."):
                data = generate_kubernetes_data(num_samples, failure_rate/100, time_steps)
                st.session_state.data = data
                st.success(f"Generated {len(data)} samples with {failure_rate}% failure rate")
    
    else:
        st.subheader("Upload Your Own Data")
        st.markdown("""
        Please upload a CSV file containing Kubernetes metrics data. 
        The file should include various metrics like CPU usage, memory usage, pod status, etc.
        It should also include a binary column named 'failure' (1 for failure, 0 for normal).
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                if 'failure' not in data.columns:
                    st.error("The uploaded data must contain a 'failure' column.")
                else:
                    st.session_state.data = data
                    st.success(f"Uploaded data with {len(data)} samples")
            except Exception as e:
                st.error(f"Error reading the file: {e}")
    
    # Display and explore the data
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head())
        
        st.subheader("Data Information")
        buffer = io.StringIO()
        st.session_state.data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Data Distribution")
        if 'failure' in st.session_state.data.columns:
            fig = px.histogram(st.session_state.data, x='failure', color='failure',
                            title='Distribution of Failure vs Non-Failure Cases')
            st.plotly_chart(fig)
        
        st.subheader("Data Preprocessing")
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                # Preprocessing
                preprocessed_data, _ = preprocess_data(st.session_state.data)
                st.session_state.preprocessed_data = preprocessed_data
                
                # Feature Engineering
                with st.expander("Feature Engineering Options"):
                    window_sizes = st.multiselect("Select window sizes for temporal features:", 
                                                [3, 5, 10, 15, 30], [5, 10])
                    use_pca = st.checkbox("Apply PCA for dimensionality reduction", True)
                    pca_components = None
                    if use_pca:
                        pca_components = st.slider("Number of PCA components", 2, 
                                                min(20, preprocessed_data.shape[1]-1), 10)
                
                feature_engineered_data = apply_feature_engineering(
                    preprocessed_data, 
                    window_sizes=window_sizes,
                    apply_pca=use_pca,
                    n_components=pca_components
                )
                st.session_state.feature_engineered_data = feature_engineered_data
                
                # Handle class imbalance
                with st.expander("Class Imbalance Handling"):
                    imbalance_method = st.selectbox(
                        "Select method to handle class imbalance:", 
                        ["None", "SMOTE", "RandomUnderSampler", "ADASYN"]
                    )
                
                # Split the data
                X = feature_engineered_data.drop('failure', axis=1)
                y = feature_engineered_data['failure']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Balance training data if method is selected
                if imbalance_method != "None":
                    X_train, y_train = handle_class_imbalance(X_train, y_train, method=imbalance_method)
                
                # Apply scaling
                with st.expander("Scaling Options"):
                    scaling_method = st.selectbox(
                        "Select scaling method:", 
                        ["StandardScaler", "RobustScaler", "MinMaxScaler"]
                    )
                
                X_train_scaled, X_test_scaled, scaler = apply_scaling(
                    X_train, X_test, method=scaling_method
                )
                
                st.session_state.X_train = X_train_scaled
                st.session_state.X_test = X_test_scaled
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                
                st.success("Data preprocessing completed!")
                
                st.subheader("Final Dataset Information")
                st.write(f"Training set shape: {X_train_scaled.shape}")
                st.write(f"Testing set shape: {X_test_scaled.shape}")
                st.write(f"Failure ratio in training set: {y_train.mean():.2%}")
                st.write(f"Failure ratio in testing set: {y_test.mean():.2%}")

# Model Training Page
elif page == "Model Training":
    st.header("2. Model Training")
    
    if st.session_state.X_train is None:
        st.warning("Please complete the data preparation step first.")
    else:
        st.subheader("Select and Train Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Random Forest Classifier")
            rf_n_estimators = st.slider("Number of estimators", 50, 300, 100)
            rf_max_depth = st.slider("Maximum depth", 5, 30, 10)
            
            if st.button("Train Random Forest"):
                with st.spinner("Training Random Forest model..."):
                    rf_model = train_random_forest(
                        st.session_state.X_train, 
                        st.session_state.y_train,
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth
                    )
                    st.session_state.models['random_forest'] = rf_model
                    
                    # Get feature importance
                    feature_importance = get_feature_importance(
                        rf_model, 
                        list(st.session_state.X_train.columns)
                    )
                    st.session_state.feature_importance = feature_importance
                    
                    # Plot feature importance
                    fig = plot_feature_importance(feature_importance, top_n=10)
                    st.pyplot(fig)
                    
                    st.success("Random Forest model trained successfully!")
        
        with col2:
            st.markdown("### Isolation Forest (Anomaly Detection)")
            if_contamination = st.slider("Contamination", 0.01, 0.5, float(st.session_state.y_train.mean()))
            if_n_estimators = st.slider("Number of estimators (IF)", 50, 300, 100)
            
            if st.button("Train Isolation Forest"):
                with st.spinner("Training Isolation Forest model..."):
                    if_model = train_isolation_forest(
                        st.session_state.X_train,
                        contamination=if_contamination,
                        n_estimators=if_n_estimators
                    )
                    st.session_state.models['isolation_forest'] = if_model
                    st.success("Isolation Forest model trained successfully!")
        
        st.markdown("### Time Series Model (ARIMA)")
        if 'timestamp' in st.session_state.data.columns:
            ts_enabled = st.checkbox("Enable Time Series Analysis", True)
            if ts_enabled:
                ts_feature = st.selectbox(
                    "Select feature for time series prediction:",
                    [col for col in st.session_state.data.columns if col not in ['failure', 'timestamp']]
                )
                if st.button("Train Time Series Model"):
                    with st.spinner("Training Time Series model..."):
                        ts_model = train_time_series_model(
                            st.session_state.data,
                            feature=ts_feature
                        )
                        st.session_state.models['time_series'] = ts_model
                        st.success("Time Series model trained successfully!")
        else:
            st.info("Time Series analysis requires a 'timestamp' column in your dataset.")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.header("3. Model Evaluation")
    
    if not st.session_state.models:
        st.warning("Please train at least one model first.")
    else:
        st.subheader("Model Performance Evaluation")
        
        available_models = list(st.session_state.models.keys())
        model_to_evaluate = st.selectbox("Select model to evaluate:", available_models)
        
        if st.button("Evaluate Model"):
            with st.spinner(f"Evaluating {model_to_evaluate} model..."):
                model = st.session_state.models[model_to_evaluate]
                
                # Evaluate the model
                if model_to_evaluate != 'time_series':
                    evaluation_results = evaluate_model(
                        model, 
                        st.session_state.X_test, 
                        st.session_state.y_test,
                        model_type=model_to_evaluate
                    )
                    st.session_state.evaluation_results[model_to_evaluate] = evaluation_results
                    
                    # Display evaluation metrics
                    st.subheader("Evaluation Metrics")
                    
                    # Extract only numeric metrics for display
                    display_metrics = {}
                    for key, value in evaluation_results.items():
                        if key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                            display_metrics[key] = value
                    
                    # Create a more user-friendly metrics display with color-coded performance indicators
                    st.write("### Key Performance Metrics")
                    
                    # Display metrics in a table format first
                    metrics_table = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
                        'Value': [
                            f"{display_metrics.get('accuracy', 0):.4f}",
                            f"{display_metrics.get('precision', 0):.4f}",
                            f"{display_metrics.get('recall', 0):.4f}",
                            f"{display_metrics.get('f1', 0):.4f}",
                            f"{display_metrics.get('auc', 0):.4f}" if 'auc' in display_metrics else 'N/A'
                        ]
                    })
                    
                    # Add interpretation column
                    metrics_table['Interpretation'] = [
                        'Excellent' if display_metrics.get('accuracy', 0) > 0.9 else 
                        'Good' if display_metrics.get('accuracy', 0) > 0.8 else 
                        'Fair' if display_metrics.get('accuracy', 0) > 0.7 else 'Poor',
                        
                        'Excellent' if display_metrics.get('precision', 0) > 0.9 else 
                        'Good' if display_metrics.get('precision', 0) > 0.8 else 
                        'Fair' if display_metrics.get('precision', 0) > 0.7 else 'Poor',
                        
                        'Excellent' if display_metrics.get('recall', 0) > 0.9 else 
                        'Good' if display_metrics.get('recall', 0) > 0.8 else 
                        'Fair' if display_metrics.get('recall', 0) > 0.7 else 'Poor',
                        
                        'Excellent' if display_metrics.get('f1', 0) > 0.9 else 
                        'Good' if display_metrics.get('f1', 0) > 0.8 else 
                        'Fair' if display_metrics.get('f1', 0) > 0.7 else 'Poor',
                        
                        'Excellent' if display_metrics.get('auc', 0) > 0.9 else 
                        'Good' if display_metrics.get('auc', 0) > 0.8 else 
                        'Fair' if display_metrics.get('auc', 0) > 0.7 else 'Poor' if 'auc' in display_metrics else 'N/A'
                    ]
                    
                    # Display the metrics table
                    st.table(metrics_table)
                    
                    # Also show metrics with visual indicators
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        accuracy = display_metrics.get('accuracy', 0)
                        st.metric("Accuracy", f"{accuracy:.4f}", 
                                 delta=f"{(accuracy-0.5):.2f} vs random" if accuracy > 0 else None)
                        
                        precision = display_metrics.get('precision', 0)
                        st.metric("Precision", f"{precision:.4f}")
                        
                        recall = display_metrics.get('recall', 0)
                        st.metric("Recall (Sensitivity)", f"{recall:.4f}")
                    
                    with col2:
                        f1 = display_metrics.get('f1', 0)
                        st.metric("F1 Score", f"{f1:.4f}")
                        
                        if 'auc' in display_metrics:
                            auc = display_metrics.get('auc', 0)
                            st.metric("AUC-ROC", f"{auc:.4f}", 
                                    delta=f"{(auc-0.5):.2f} vs random" if auc > 0.5 else None)
                        
                        # Calculate specificity if available in the report
                        if 'classification_report' in evaluation_results:
                            try:
                                report = evaluation_results['classification_report']
                                if '0' in report:  # Class 0 exists in report
                                    specificity = report['0']['recall']
                                    st.metric("Specificity", f"{specificity:.4f}")
                            except:
                                pass
                    
                    # Display detailed classification report
                    if 'classification_report' in evaluation_results:
                        with st.expander("View Detailed Classification Report"):
                            try:
                                report = evaluation_results['classification_report']
                                # Convert the classification report dict to a DataFrame
                                report_df = pd.DataFrame(report).T
                                # Filter out unnecessary rows
                                if 'accuracy' in report_df.index:
                                    report_df = report_df.drop(['accuracy'])
                                # Rename index
                                report_df.index.name = 'Class'
                                report_df = report_df.rename(index={'0': 'Normal (0)', '1': 'Failure (1)'})
                                # Display the report
                                st.dataframe(report_df.style.format("{:.4f}"))
                                
                                # Add interpretation
                                st.markdown("""
                                **Interpretation Guide:**
                                - **Precision**: Percentage of correctly identified failures among all predictions.
                                - **Recall**: Percentage of actual failures correctly identified (also called sensitivity).
                                - **F1-score**: Harmonic mean of precision and recall (balance between the two).
                                - **Support**: Number of samples in each class.
                                """)
                            except Exception as e:
                                st.error(f"Error displaying classification report: {e}")
                    
                    # Display additional metrics in a dataframe
                    with st.expander("View All Performance Metrics"):
                        # Filter out array metrics and already displayed metrics
                        scalar_metrics = {}
                        exclude_keys = ['fpr', 'tpr', 'predictions', 'probabilities', 'anomaly_scores', 
                                      'classification_report', 'accuracy', 'precision', 'recall', 'f1', 'auc']
                        
                        for key, value in evaluation_results.items():
                            if key not in exclude_keys:
                                try:
                                    # Check if value is scalar or convertible to scalar
                                    float(value)
                                    scalar_metrics[key] = value
                                except (TypeError, ValueError):
                                    pass
                        
                        if scalar_metrics:
                            metrics_df = pd.DataFrame({
                                'Metric': list(scalar_metrics.keys()),
                                'Value': list(scalar_metrics.values())
                            })
                            st.dataframe(metrics_df)
                    
                    # Plot confusion matrix
                    if model_to_evaluate == 'random_forest':
                        st.subheader("Confusion Matrix")
                        fig = plot_confusion_matrix(
                            st.session_state.y_test,
                            evaluation_results['predictions']
                        )
                        st.pyplot(fig)
                    
                    # Plot ROC curve if available
                    if 'fpr' in evaluation_results and 'tpr' in evaluation_results:
                        st.subheader("ROC Curve")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=evaluation_results['fpr'], 
                            y=evaluation_results['tpr'],
                            mode='lines',
                            name=f'ROC (AUC = {evaluation_results["auc"]:.3f})'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            line=dict(dash='dash', width=1),
                            name='Random'
                        ))
                        fig.update_layout(
                            title='Receiver Operating Characteristic (ROC) Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            height=500,
                            width=700
                        )
                        st.plotly_chart(fig)
                    
                    # For Isolation Forest, visualize anomaly detection
                    if model_to_evaluate == 'isolation_forest':
                        st.subheader("Anomaly Detection Visualization")
                        fig = plot_anomaly_detection(
                            st.session_state.X_test, 
                            evaluation_results['predictions']
                        )
                        st.plotly_chart(fig)
                
                else:  # Time Series model visualization
                    st.subheader("Time Series Forecast")
                    # Plot time series predictions
                    fig = plot_metrics_over_time(
                        st.session_state.data, 
                        model
                    )
                    st.plotly_chart(fig)
                
        # Compare models if multiple models are trained
        if len(st.session_state.evaluation_results) > 1:
            st.subheader("Model Comparison")
            
            comparison_metrics = ['accuracy', 'precision', 'recall', 'f1']
            comparison_data = {}
            
            for model_name, results in st.session_state.evaluation_results.items():
                if model_name != 'time_series':  # Skip time series for comparison
                    comparison_data[model_name] = {
                        metric: results.get(metric, 0) 
                        for metric in comparison_metrics if metric in results
                    }
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Plot comparison
                fig = px.bar(
                    comparison_df.reset_index().melt(id_vars='index'),
                    x='index',
                    y='value',
                    color='variable',
                    barmode='group',
                    title='Model Comparison',
                    labels={'index': 'Metric', 'value': 'Score', 'variable': 'Model'}
                )
                st.plotly_chart(fig)

# Prediction Page
elif page == "Prediction":
    st.header("4. Make Predictions")
    
    if not st.session_state.models:
        st.warning("Please train at least one model first.")
    else:
        st.subheader("Input Cluster Metrics")
        
        input_method = st.radio("Select input method:", ["Manual Input", "Upload Test Data"])
        
        if input_method == "Manual Input":
            # Create form for manual input
            with st.form("prediction_form"):
                st.markdown("Enter Kubernetes cluster metrics:")
                
                # Get feature names from training data
                if st.session_state.X_train is not None:
                    features = st.session_state.X_train.columns
                    input_values = {}
                    
                    # Create two columns for better layout
                    col1, col2 = st.columns(2)
                    
                    # Display half of the features in each column
                    half = len(features) // 2
                    
                    with col1:
                        for feature in features[:half]:
                            # Use sensible defaults for each feature
                            default_val = 0.5
                            input_values[feature] = st.number_input(
                                f"{feature}:", 
                                value=default_val,
                                step=0.1
                            )
                    
                    with col2:
                        for feature in features[half:]:
                            default_val = 0.5
                            input_values[feature] = st.number_input(
                                f"{feature}:", 
                                value=default_val,
                                step=0.1
                            )
                    
                    submitted = st.form_submit_button("Predict")
                    
                    # Initialize results outside the conditional block
                    results = {}
                    
                    if submitted:
                        # Convert inputs to DataFrame
                        input_df = pd.DataFrame([input_values])
                        
                        # Make predictions with all models
                        for model_name, model in st.session_state.models.items():
                            if model_name != 'time_series':  # Skip time series model for single input
                                try:
                                    if model_name == 'random_forest':
                                        pred_proba = model.predict_proba(input_df)[0][1]
                                        prediction = model.predict(input_df)[0]
                                        results[model_name] = {
                                            'prediction': 'Failure' if prediction == 1 else 'Normal',
                                            'probability': f"{pred_proba:.2%}"
                                        }
                                    elif model_name == 'isolation_forest':
                                        # For isolation forest, -1 is anomaly, 1 is normal
                                        anomaly_score = model.score_samples(input_df)[0]
                                        prediction = model.predict(input_df)[0]
                                        results[model_name] = {
                                            'prediction': 'Anomaly' if prediction == -1 else 'Normal',
                                            'anomaly_score': f"{anomaly_score:.4f}"
                                        }
                                except Exception as e:
                                    results[model_name] = {
                                        'error': str(e)
                                    }
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("Prediction Results")
                        
                        # Simple table display for all results
                        table_data = []
                        
                        for model_name, result in results.items():
                            if 'error' in result:
                                row = {
                                    'Model': model_name.replace('_', ' ').title(),
                                    'Prediction': "Error",
                                    'Details': result['error']
                                }
                            else:
                                row = {
                                    'Model': model_name.replace('_', ' ').title(),
                                    'Prediction': result.get('prediction', 'N/A')
                                }
                                if 'probability' in result:
                                    row['Probability'] = result['probability']
                                if 'anomaly_score' in result:
                                    row['Anomaly Score'] = result['anomaly_score']
                            table_data.append(row)
                        
                        # Convert to DataFrame and display as table
                        if table_data:
                            results_df = pd.DataFrame(table_data)
                            st.table(results_df)  # Using st.table for fixed-width display
                        
                        # Display model performance metrics
                        if st.session_state.evaluation_results:
                            st.markdown("### Model Performance Metrics")
                            perf_metrics = ['accuracy', 'precision', 'recall', 'f1']
                            
                            # Create performance metrics table
                            perf_data = []
                            for model_name, eval_results in st.session_state.evaluation_results.items():
                                if model_name != 'time_series':
                                    row = {'Model': model_name.replace('_', ' ').title()}
                                    for metric in perf_metrics:
                                        if metric in eval_results:
                                            row[metric.capitalize()] = f"{eval_results[metric]:.4f}"
                                    perf_data.append(row)
                            
                            if perf_data:
                                perf_df = pd.DataFrame(perf_data)
                                st.table(perf_df)
                        
                        # Show visual indicators for each model prediction
                        if results:
                            st.markdown("### Prediction Details")
                            
                            for model_name, result in results.items():
                                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                                if 'error' in result:
                                    st.error(f"Error: {result['error']}")
                                else:
                                    if 'prediction' in result:
                                        prediction = result['prediction']
                                        is_failure = prediction in ['Failure', 'Anomaly']
                                        
                                        # Create columns for better layout
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            if is_failure:
                                                st.error(f"Prediction: {prediction}")
                                            else:
                                                st.success(f"Prediction: {prediction}")
                                        
                                        with col2:
                                            if 'probability' in result:
                                                st.metric("Failure Probability", result['probability'])
                                            if 'anomaly_score' in result:
                                                st.metric("Anomaly Score", result['anomaly_score'])
                                        
                                        # Add more context to the prediction
                                        if is_failure:
                                            st.warning("⚠️ Potential cluster issue detected! Recommendation: Review system resources and logs for irregularities.")
                                        else:
                                            st.info("✅ Cluster appears to be operating normally based on the current metrics.")
                else:
                    st.error("No trained models available. Please complete the model training step first.")
        
        else:  # Upload Test Data
            st.markdown("Upload a CSV file with test data:")
            test_file = st.file_uploader("Choose a CSV file for batch prediction", type="csv")
            
            if test_file is not None:
                try:
                    test_data = pd.read_csv(test_file)
                    st.dataframe(test_data.head())
                    
                    # Check if the test data has the same features as training data
                    if st.session_state.X_train is not None:
                        training_cols = set(st.session_state.X_train.columns)
                        test_cols = set(test_data.columns)
                        
                        # Find missing columns
                        missing_cols = training_cols - test_cols
                        
                        if missing_cols:
                            st.warning(f"Test data is missing these columns: {missing_cols}")
                        else:
                            # Keep only the columns used in training
                            test_data_processed = test_data[st.session_state.X_train.columns]
                            
                            # Make predictions with all models
                            if st.button("Run Batch Prediction"):
                                with st.spinner("Making predictions..."):
                                    results = {}
                                    
                                    for model_name, model in st.session_state.models.items():
                                        if model_name != 'time_series':
                                            try:
                                                if model_name == 'random_forest':
                                                    predictions = model.predict(test_data_processed)
                                                    probabilities = model.predict_proba(test_data_processed)[:, 1]
                                                    
                                                    results[model_name] = {
                                                        'predictions': predictions,
                                                        'probabilities': probabilities
                                                    }
                                                elif model_name == 'isolation_forest':
                                                    predictions = model.predict(test_data_processed)
                                                    scores = model.score_samples(test_data_processed)
                                                    
                                                    # Convert -1/1 to 1/0 (anomaly/normal)
                                                    predictions = np.where(predictions == -1, 1, 0)
                                                    
                                                    results[model_name] = {
                                                        'predictions': predictions,
                                                        'scores': scores
                                                    }
                                            except Exception as e:
                                                st.error(f"Error with {model_name}: {e}")
                                    
                                    # Display results
                                    st.subheader("Batch Prediction Results")
                                    
                                    # First show a summary of all model results
                                    summary_metrics = {}
                                    
                                    for model_name, result in results.items():
                                        if 'predictions' in result:
                                            failure_count = int(sum(result['predictions']))
                                            total_count = len(result['predictions'])
                                            failure_percentage = (failure_count / total_count) * 100
                                            
                                            summary_metrics[model_name] = {
                                                'Total Samples': total_count,
                                                'Predicted Failures': failure_count,
                                                'Failure Rate': f"{failure_percentage:.2f}%"
                                            }
                                    
                                    if summary_metrics:
                                        # Convert to DataFrame for display
                                        summary_df = pd.DataFrame(summary_metrics).T
                                        summary_df.index.name = 'Model'
                                        summary_df = summary_df.reset_index()
                                        
                                        # Display summary metrics
                                        st.write("### Prediction Summary")
                                        st.dataframe(summary_df)
                                        
                                        # Create a bar chart comparing failure rates
                                        fig = px.bar(
                                            summary_df, 
                                            x='Model', 
                                            y='Predicted Failures',
                                            title='Predicted Failures by Model',
                                            color='Model'
                                        )
                                        st.plotly_chart(fig)
                                    
                                    # Show detailed results for each model
                                    for model_name, result in results.items():
                                        with st.expander(f"{model_name.replace('_', ' ').title()} Detailed Predictions"):
                                            if 'predictions' in result:
                                                # Add predictions to a copy of the original data
                                                result_df = test_data.copy()
                                                result_df[f'{model_name}_prediction'] = result['predictions']
                                                
                                                if 'probabilities' in result:
                                                    result_df[f'{model_name}_probability'] = result['probabilities']
                                                if 'scores' in result:
                                                    result_df[f'{model_name}_anomaly_score'] = result['scores']
                                                
                                                # Display the dataframe with predictions
                                                st.dataframe(result_df)
                                                
                                                # Calculate and display metrics
                                                failure_count = sum(result['predictions'])
                                                total_count = len(result['predictions'])
                                                failure_percentage = (failure_count / total_count) * 100
                                                
                                                # Show metrics in a more visual way
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Total Samples", total_count)
                                                with col2:
                                                    st.metric("Predicted Failures", int(failure_count))
                                                with col3:
                                                    st.metric("Failure Rate", f"{failure_percentage:.2f}%")
                                                
                                                # Plot predictions distribution
                                                fig = px.histogram(
                                                    result_df, 
                                                    x=f'{model_name}_prediction',
                                                    color=f'{model_name}_prediction',
                                                    title=f'Distribution of Predictions - {model_name.replace("_", " ").title()}'
                                                )
                                                st.plotly_chart(fig)
                                
                                st.success("Batch prediction completed!")
                    else:
                        st.error("No trained models available. Please complete the model training step first.")
                
                except Exception as e:
                    st.error(f"Error reading the file: {e}")

# Add footer
st.markdown("---")
st.markdown("""
**About this application:**  
This application uses machine learning to predict failures in Kubernetes clusters based on various metrics. 
It supports data preprocessing, feature engineering, model training, evaluation, and making predictions.
""")
