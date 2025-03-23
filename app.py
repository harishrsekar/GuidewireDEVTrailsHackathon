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
                        from utils import format_prediction_result
                        
                        results = {
                            'status': 'success',
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'predictions': {},
                            'metadata': {
                                'feature_count': len(input_df.columns),
                                'models_available': list(st.session_state.models.keys()),
                                'request_id': f"req_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                                'input_type': 'manual'
                            }
                        }

                        for model_name, model in st.session_state.models.items():
                            if model_name != 'time_series':  # Skip time series model for single input
                                try:
                                    if model_name == 'random_forest':
                                        pred_proba = model.predict_proba(input_df)[0][1]
                                        prediction = model.predict(input_df)[0]
                                        prediction_data = {
                                            'model_type': model_name,
                                            'prediction_time': pd.Timestamp.now().isoformat(),
                                            'prediction': 'Failure' if prediction == 1 else 'Normal',
                                            'probability': float(pred_proba),
                                            'probability_formatted': f"{pred_proba:.2%}",
                                            'confidence_level': 'High' if abs(pred_proba - 0.5) > 0.3 else 'Medium' if abs(pred_proba - 0.5) > 0.15 else 'Low'
                                        }
                                    elif model_name == 'isolation_forest':
                                        anomaly_score = model.score_samples(input_df)[0]
                                        prediction = model.predict(input_df)[0]
                                        prediction_data = {
                                            'model_type': model_name,
                                            'prediction_time': pd.Timestamp.now().isoformat(),
                                            'prediction': 'Anomaly' if prediction == -1 else 'Normal',
                                            'anomaly_score': float(anomaly_score),
                                            'anomaly_score_formatted': f"{anomaly_score:.4f}",
                                            'severity': 'High' if abs(anomaly_score) > 0.75 else 'Medium' if abs(anomaly_score) > 0.5 else 'Low'
                                        }
                                        
                                    # Get performance metrics from evaluation if available
                                    performance_metrics = None
                                    if model_name in st.session_state.evaluation_results:
                                        eval_results = st.session_state.evaluation_results[model_name]
                                        performance_metrics = {
                                            'accuracy': eval_results.get('accuracy', 0),
                                            'precision': eval_results.get('precision', 0),
                                            'recall': eval_results.get('recall', 0),
                                            'f1': eval_results.get('f1', 0),
                                            'auc': eval_results.get('auc', 0) if 'auc' in eval_results else None
                                        }
                                    
                                    # Use the utility function to format the results consistently
                                    results['predictions'][model_name] = format_prediction_result(
                                        model_name, 
                                        prediction_data,
                                        performance_metrics=performance_metrics
                                    )

                                except Exception as e:
                                    import traceback
                                    error_details = {
                                        'error_message': str(e),
                                        'traceback': traceback.format_exc(),
                                        'error_type': type(e).__name__
                                    }
                                    results['predictions'][model_name] = format_prediction_result(
                                        model_name, 
                                        prediction_data={},
                                        error=str(e)
                                    )

                        # Display results
                        st.markdown("---")
                        st.subheader("Prediction Results")

                        if results['status'] == 'success':
                            st.success("✅ Predictions completed successfully")

                            # Display metadata
                            with st.expander("Prediction Metadata", expanded=False):
                                st.json(results['metadata'])

                            # Create tabs for different visualization options
                            tab1, tab2, tab3 = st.tabs(["Summary View", "Detailed View", "Performance Metrics"])
                            
                            # Initialize session state for tab selection if needed
                            if 'selected_tab' not in st.session_state:
                                st.session_state.selected_tab = 0  # Default to Summary View

                            with tab1:
                                st.markdown("### Prediction Summary")
                                table_data = []
                                
                                # Process all model predictions for the table
                                for model_name, result in results['predictions'].items():
                                    if result['status'] == 'error':
                                        row = {
                                            'Model': model_name.replace('_', ' ').title(),
                                            'Status': '❌ Error',
                                            'Details': result.get('error_message', result.get('error', 'Unknown error'))
                                        }
                                    else:
                                        row = {
                                            'Model': model_name.replace('_', ' ').title(),
                                            'Status': '✅ Success',
                                            'Prediction': result['prediction']
                                        }
                                        
                                        # Add model-specific metrics
                                        if 'probability_formatted' in result:
                                            row['Probability'] = result['probability_formatted']
                                            row['Confidence'] = result['confidence_level']
                                        if 'anomaly_score_formatted' in result:
                                            row['Anomaly Score'] = result['anomaly_score_formatted']
                                            row['Severity'] = result['severity']
                                    
                                    table_data.append(row)
                                
                                # Display the summary table
                                if table_data:
                                    results_df = pd.DataFrame(table_data)
                                    
                                    # Reorder columns to show important info first
                                    # First the key columns
                                    cols = ['Model', 'Status', 'Prediction']
                                    
                                    # Then prediction details
                                    pred_cols = ['Probability', 'Confidence', 'Anomaly Score', 'Severity']
                                    for col in pred_cols:
                                        if col in results_df.columns:
                                            cols.append(col)
                                    
                                    # Add any remaining columns
                                    cols.extend([col for col in results_df.columns if col not in cols])
                                    
                                    # Apply the column ordering
                                    results_df = results_df[cols]
                                    
                                    st.table(results_df)  # Using st.table for fixed-width display
                                
                                # Show visual indicators for each model
                                st.markdown("### Prediction Details")
                                for model_name, result in results['predictions'].items():
                                    if result['status'] == 'success':
                                        st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            prediction = result['prediction']
                                            if prediction in ['Failure', 'Anomaly']:
                                                st.error(f"⚠️ {prediction} Detected")
                                            else:
                                                st.success(f"✅ System Normal")
                                        
                                        with col2:
                                            if 'probability_formatted' in result:
                                                st.metric("Failure Probability", 
                                                         result['probability_formatted'],
                                                         help=f"Confidence: {result['confidence_level']}")
                                            if 'anomaly_score_formatted' in result:
                                                st.metric("Anomaly Score", 
                                                         result['anomaly_score_formatted'],
                                                         help=f"Severity: {result['severity']}")
                            
                            with tab2:
                                # Show raw JSON data in an expandable container
                                with st.expander("Raw Prediction Data", expanded=True):
                                    st.json(results)

                                # Display recommendations based on predictions
                                st.subheader("Recommendations")
                                for model_name, result in results['predictions'].items():
                                    if result['status'] == 'success':
                                        prediction = result['prediction']
                                        if prediction in ['Failure', 'Anomaly']:
                                            st.warning(f"""
                                            **{model_name} Recommendations:**
                                            1. Review system resources and logs
                                            2. Check for resource constraints
                                            3. Monitor affected components
                                            4. Consider scaling resources if needed
                                            """)
                                        else:
                                            st.info(f"""
                                            **{model_name} Status:**
                                            - System operating normally
                                            - Continue regular monitoring
                                            """)
                            
                            with tab3:
                                st.markdown("### Model Performance Metrics")
                                
                                # Create a table for performance metrics
                                perf_data = []
                                
                                # Add metrics from prediction results if available
                                for model_name, result in results['predictions'].items():
                                    if result['status'] == 'success' and 'performance_metrics' in result and result['performance_metrics']:
                                        metrics = result['performance_metrics']
                                        row = {
                                            'Model': model_name.replace('_', ' ').title(),
                                            'Accuracy': f"{metrics['accuracy']:.4f}",
                                            'Precision': f"{metrics['precision']:.4f}",
                                            'Recall': f"{metrics['recall']:.4f}",
                                            'F1 Score': f"{metrics['f1']:.4f}"
                                        }
                                        if metrics.get('auc') is not None:
                                            row['AUC'] = f"{metrics['auc']:.4f}"
                                        perf_data.append(row)
                                
                                # Also add metrics from evaluation results if available
                                if not perf_data and st.session_state.evaluation_results:
                                    for model_name, eval_results in st.session_state.evaluation_results.items():
                                        if model_name != 'time_series':
                                            row = {'Model': model_name.replace('_', ' ').title()}
                                            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                                                if metric in eval_results:
                                                    row[metric.capitalize()] = f"{eval_results[metric]:.4f}"
                                            if 'auc' in eval_results:
                                                row['AUC'] = f"{eval_results['auc']:.4f}"
                                            perf_data.append(row)
                                
                                # Display performance metrics table
                                if perf_data:
                                    perf_df = pd.DataFrame(perf_data)
                                    st.table(perf_df)
                                    
                                    # Add interactive model performance comparison slider
                                    st.subheader("Interactive Model Performance Comparison")
                                    
                                    if len(perf_data) > 1:  # Only show comparison if we have multiple models
                                        # Convert string metrics to float for comparison
                                        numeric_df = perf_df.copy()
                                        for col in numeric_df.columns:
                                            if col != 'Model':
                                                try:
                                                    numeric_df[col] = numeric_df[col].astype(float)
                                                except:
                                                    pass  # Skip if can't convert
                                        
                                        # Get available metrics (excluding the Model column)
                                        available_metrics = [col for col in numeric_df.columns if col != 'Model']
                                        
                                        if available_metrics:
                                            # Let user select which metric to compare
                                            selected_metric = st.selectbox(
                                                "Select metric to compare across models:",
                                                options=available_metrics,
                                                index=0  # Default to first metric (likely accuracy)
                                            )
                                            
                                            # Create a slider to set the threshold for the selected metric
                                            threshold = st.slider(
                                                f"Set minimum threshold for {selected_metric}",
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=0.7,  # Default threshold
                                                step=0.01
                                            )
                                            
                                            # Filter models that meet the threshold
                                            models_above_threshold = numeric_df[numeric_df[selected_metric] >= threshold]
                                            models_below_threshold = numeric_df[numeric_df[selected_metric] < threshold]
                                            
                                            # Display results
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.markdown(f"### Models Above {selected_metric} = {threshold:.2f}")
                                                if not models_above_threshold.empty:
                                                    st.dataframe(models_above_threshold[['Model', selected_metric]])
                                                else:
                                                    st.info(f"No models meet the {selected_metric} threshold of {threshold:.2f}")
                                            
                                            with col2:
                                                st.markdown(f"### Models Below {selected_metric} = {threshold:.2f}")
                                                if not models_below_threshold.empty:
                                                    st.dataframe(models_below_threshold[['Model', selected_metric]])
                                                else:
                                                    st.success(f"All models meet the {selected_metric} threshold of {threshold:.2f}")
                                            
                                            # Create a horizontal bar chart comparing models on the selected metric
                                            chart_data = numeric_df.sort_values(by=selected_metric, ascending=True)
                                            
                                            # Assign colors based on threshold
                                            colors = ['green' if val >= threshold else 'red' for val in chart_data[selected_metric]]
                                            
                                            # Create a plotly figure for better interactivity
                                            fig = px.bar(
                                                chart_data,
                                                x=selected_metric, 
                                                y='Model',
                                                orientation='h',
                                                color=selected_metric,
                                                color_continuous_scale=['red', 'yellow', 'green'],
                                                range_color=[0, 1],
                                                labels={selected_metric: selected_metric, 'Model': 'Model'},
                                                title=f'{selected_metric} Comparison Across Models',
                                                text=chart_data[selected_metric].apply(lambda x: f'{x:.4f}')
                                            )
                                            
                                            # Add a vertical line for the threshold
                                            fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                                                        annotation_text=f"Threshold: {threshold:.2f}", 
                                                        annotation_position="top right")
                                            
                                            # Improve layout
                                            fig.update_layout(
                                                height=400,
                                                xaxis_range=[0, 1.1],
                                                xaxis_title=selected_metric,
                                                yaxis_title='Model',
                                                showlegend=False
                                            )
                                            
                                            # Format text display
                                            fig.update_traces(textposition='outside')
                                            
                                            # Display the interactive Plotly chart
                                            st.plotly_chart(fig)
                                            
                                            # Add an explanation about comparing metrics
                                            with st.expander("About Metric Comparison"):
                                                st.markdown(f"""
                                                **Understanding {selected_metric} Comparison:**
                                                
                                                - **Green bars** represent models that meet or exceed your threshold of {threshold:.2f}
                                                - **Red bars** represent models that fall below your threshold
                                                - Use the slider to adjust the threshold and see which models qualify
                                                - Higher {selected_metric} generally indicates better model performance, but always consider your specific use case
                                                
                                                When comparing models, it's important to look at multiple metrics, not just {selected_metric}. 
                                                Some models might excel in one metric but perform poorly in others.
                                                """)
                                    else:
                                        st.info("Multiple models required for comparison. Train and evaluate at least two different models.")
                                    
                                    # Add explanation of metrics
                                    with st.expander("Understanding Performance Metrics"):
                                        st.markdown("""
                                        **Accuracy**: Percentage of correct predictions (TP + TN) / (TP + TN + FP + FN)
                                        
                                        **Precision**: Proportion of positive identifications that were actually correct. TP / (TP + FP)
                                        
                                        **Recall**: Proportion of actual positives that were identified correctly. TP / (TP + FN)
                                        
                                        **F1 Score**: Harmonic mean of precision and recall. 2 * (Precision * Recall) / (Precision + Recall)
                                        
                                        **AUC**: Area Under the ROC Curve. Measures the ability to distinguish between classes.
                                        
                                        Where: TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives
                                        """)
                                    
                                    # Add confusion matrix visualization for any available model
                                    st.subheader("Confusion Matrix")
                                    
                                    # Allow user to select which model's confusion matrix to view
                                    models_with_cm = []
                                    
                                    # Check which models have confusion matrices available
                                    for model_name in st.session_state.models:
                                        if (model_name in st.session_state.evaluation_results and 
                                            'confusion_matrix' in st.session_state.evaluation_results[model_name] and
                                            model_name != 'time_series'):  # Exclude time series model
                                            models_with_cm.append(model_name)
                                    
                                    if models_with_cm:
                                        selected_model = st.selectbox(
                                            "Select model for confusion matrix visualization",
                                            options=models_with_cm,
                                            format_func=lambda x: x.replace('_', ' ').title()
                                        )
                                        
                                        # Display the confusion matrix for the selected model
                                        from visualizer import plot_confusion_matrix
                                        
                                        # Use the stored confusion matrix from evaluation results
                                        cm = st.session_state.evaluation_results[selected_model]['confusion_matrix']
                                        
                                        # Display information about the confusion matrix
                                        st.markdown("""
                                        **What This Shows:**
                                        - **Bottom Right:** True Positives - Correctly predicted failures
                                        - **Bottom Left:** False Negatives - Failures missed by the model
                                        - **Top Right:** False Positives - False alarms
                                        - **Top Left:** True Negatives - Correctly predicted normal operation
                                        """)
                                        
                                        # Plot using the stored confusion matrix
                                        cm_fig = plot_confusion_matrix(None, None, cm=cm)
                                        st.pyplot(cm_fig)
                                        
                                        # Add explanation of confusion matrix
                                        with st.expander("Understanding Confusion Matrix"):
                                            st.markdown("""
                                            **Confusion Matrix Explained:**
                                            
                                            - **True Negative (Top-Left)**: Normal instances correctly predicted as normal
                                            - **False Positive (Top-Right)**: Normal instances incorrectly predicted as failures
                                            - **False Negative (Bottom-Left)**: Failure instances incorrectly predicted as normal
                                            - **True Positive (Bottom-Right)**: Failure instances correctly predicted as failures
                                            
                                            This visualization helps identify which types of errors the model tends to make.
                                            """)
                                    else:
                                        st.info("No confusion matrix data available. Complete model evaluation to see confusion matrix.")
                                        
                                else:
                                    st.info("No performance metrics available. Complete model evaluation to see metrics.")

                        # Performance metrics are now displayed in the tabbed interface

                        # Visual indicators are now handled in the tabbed interface above
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
                                    from utils import format_prediction_result
                                    
                                    # Create a structured response format
                                    results = {
                                        'status': 'success',
                                        'timestamp': pd.Timestamp.now().isoformat(),
                                        'predictions': {},
                                        'metadata': {
                                            'feature_count': len(test_data_processed.columns),
                                            'sample_count': len(test_data_processed),
                                            'models_available': list(st.session_state.models.keys()),
                                            'request_id': f"batch_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                                            'input_type': 'batch'
                                        }
                                    }

                                    for model_name, model in st.session_state.models.items():
                                        if model_name != 'time_series':  # Skip time series model for batch
                                            try:
                                                if model_name == 'random_forest':
                                                    predictions = model.predict(test_data_processed)
                                                    probabilities = model.predict_proba(test_data_processed)[:, 1]
                                                    prediction_data = {
                                                        'model_type': model_name,
                                                        'prediction_time': pd.Timestamp.now().isoformat(),
                                                        'predictions': predictions.tolist(),  # Convert to list for JSON serialization
                                                        'probabilities': probabilities.tolist(), # Convert to list for JSON serialization
                                                        'prediction_summary': {
                                                            'failure_count': int(sum(predictions)),
                                                            'total_count': len(predictions),
                                                            'failure_rate': float(sum(predictions) / len(predictions))
                                                        }
                                                    }
                                                    
                                                elif model_name == 'isolation_forest':
                                                    raw_predictions = model.predict(test_data_processed)
                                                    scores = model.score_samples(test_data_processed)
                                                    # Convert -1/1 to 1/0 (anomaly/normal)
                                                    predictions = np.where(raw_predictions == -1, 1, 0)
                                                    prediction_data = {
                                                        'model_type': model_name,
                                                        'prediction_time': pd.Timestamp.now().isoformat(),
                                                        'predictions': predictions.tolist(),  # Convert to list for JSON serialization
                                                        'scores': scores.tolist(),  # Convert to list for JSON serialization
                                                        'prediction_summary': {
                                                            'anomaly_count': int(sum(predictions)),
                                                            'total_count': len(predictions),
                                                            'anomaly_rate': float(sum(predictions) / len(predictions))
                                                        }
                                                    }
                                                
                                                # Get performance metrics from evaluation if available
                                                performance_metrics = None
                                                if model_name in st.session_state.evaluation_results:
                                                    eval_results = st.session_state.evaluation_results[model_name]
                                                    performance_metrics = {
                                                        'accuracy': eval_results.get('accuracy', 0),
                                                        'precision': eval_results.get('precision', 0),
                                                        'recall': eval_results.get('recall', 0),
                                                        'f1': eval_results.get('f1', 0),
                                                        'auc': eval_results.get('auc', 0) if 'auc' in eval_results else None
                                                    }
                                                
                                                # Use the utility function to format the results consistently
                                                results['predictions'][model_name] = format_prediction_result(
                                                    model_name, 
                                                    prediction_data,
                                                    performance_metrics=performance_metrics
                                                )
                                                
                                            except Exception as e:
                                                import traceback
                                                error_details = {
                                                    'error_message': str(e),
                                                    'traceback': traceback.format_exc(),
                                                    'error_type': type(e).__name__
                                                }
                                                results['predictions'][model_name] = format_prediction_result(
                                                    model_name, 
                                                    prediction_data={},
                                                    error=str(e)
                                                )
                                                st.error(f"Error with {model_name}: {e}")

                                    # Display results
                                    st.subheader("Batch Prediction Results")

                                    # First show a summary of all model results
                                    summary_metrics = {}

                                    for model_name, result in results['predictions'].items():
                                        if result['status'] == 'success' and 'prediction_summary' in result:
                                            summary = result['prediction_summary']
                                            
                                            # Display different metrics based on model type
                                            if result['model_type'] == 'random_forest':
                                                failure_count = summary['failure_count']
                                                total_count = summary['total_count']
                                                failure_rate = summary['failure_rate']
                                                
                                                summary_metrics[model_name] = {
                                                    'Total Samples': total_count,
                                                    'Predicted Failures': failure_count,
                                                    'Failure Rate': f"{failure_rate * 100:.2f}%"
                                                }
                                            elif result['model_type'] == 'isolation_forest':
                                                anomaly_count = summary['anomaly_count']
                                                total_count = summary['total_count']
                                                anomaly_rate = summary['anomaly_rate']
                                                
                                                summary_metrics[model_name] = {
                                                    'Total Samples': total_count,
                                                    'Detected Anomalies': anomaly_count,
                                                    'Anomaly Rate': f"{anomaly_rate * 100:.2f}%"
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
                                    for model_name, result in results['predictions'].items():
                                        with st.expander(f"{model_name.replace('_', ' ').title()} Detailed Predictions"):
                                            if result['status'] == 'success':
                                                # Add predictions to a copy of the original data
                                                result_df = test_data.copy()
                                                
                                                if 'predictions' in result:
                                                    result_df[f'{model_name}_prediction'] = result['predictions']
                                                else:
                                                    # Try the model-specific location for predictions
                                                    result_df[f'{model_name}_prediction'] = result.get('prediction_summary', {}).get('predictions', [])
                                                
                                                if 'probabilities' in result:
                                                    result_df[f'{model_name}_probability'] = result['probabilities']
                                                if 'scores' in result:
                                                    result_df[f'{model_name}_anomaly_score'] = result['scores']

                                                # Display the dataframe with predictions
                                                st.dataframe(result_df)

                                                # Get metrics from the prediction summary if available
                                                if 'prediction_summary' in result:
                                                    summary = result['prediction_summary']
                                                    
                                                    # Show metrics in a more visual way
                                                    col1, col2, col3 = st.columns(3)
                                                    
                                                    with col1:
                                                        st.metric("Total Samples", summary.get('total_count', 0))
                                                    
                                                    # Display different metrics based on model type
                                                    if result['model_type'] == 'random_forest':
                                                        with col2:
                                                            st.metric("Predicted Failures", int(summary.get('failure_count', 0)))
                                                        with col3:
                                                            failure_rate = summary.get('failure_rate', 0)
                                                            st.metric("Failure Rate", f"{failure_rate * 100:.2f}%")
                                                    elif result['model_type'] == 'isolation_forest':
                                                        with col2:
                                                            st.metric("Detected Anomalies", int(summary.get('anomaly_count', 0)))
                                                        with col3:
                                                            anomaly_rate = summary.get('anomaly_rate', 0)
                                                            st.metric("Anomaly Rate", f"{anomaly_rate * 100:.2f}%")
                                                
                                                # Plot predictions distribution
                                                if f'{model_name}_prediction' in result_df.columns:
                                                    fig = px.histogram(
                                                        result_df, 
                                                        x=f'{model_name}_prediction',
                                                        color=f'{model_name}_prediction',
                                                        title=f'Distribution of Predictions - {model_name.replace("_", " ").title()}'
                                                    )
                                                    st.plotly_chart(fig)
                                            else:
                                                # Display error message
                                                st.error(f"Error during prediction: {result.get('error', 'Unknown error')}")

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