import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc,
                           classification_report)

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
