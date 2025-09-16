import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import os

def train_gbt_model(gold_data_path, model_output_path, metrics_output_path):
    """
    Trains a Gradient Boosting Classifier model using scikit-learn.
    
    Args:
        gold_data_path (str): Path to the final, processed Gold layer data.
        model_output_path (str): Path to save the trained model.
        metrics_output_path (str): Path to save the performance metrics.
    """
    print("--- Starting scikit-learn model training ---")

    # Load the final, prepared data from the Gold layer
    gold_data = pd.read_parquet(gold_data_path)
    
    # Separate features and target
    X = gold_data.drop(['loan_application_id', 'label'], axis=1)
    y = gold_data['label']

    # Temporal split is not a direct feature of scikit-learn's basic split,
    # so we'll use a standard train-test split for simplicity.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model Training with a simple grid search ---
    # We'll stick to a simple model and then discuss hyperparameter tuning later if needed.
    
    # Define the model with a simple set of parameters
    gbt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )

    # Fit the model
    gbt.fit(X_train, y_train)
    
    print("--- Model training complete ---")

    # --- Score and Evaluate ---
    y_pred = gbt.predict(X_test)
    y_proba = gbt.predict_proba(X_test)[:, 1]

    # Calculate metrics
    auroc = roc_auc_score(y_test, y_proba)
    aupr = average_precision_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Feature Importances ---
    feature_importances = gbt.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(feature_importances)[::-1]
    
    # Create a list of top 25 feature importances
    top_25_importances = []
    for i in sorted_indices[:25]:
        top_25_importances.append({
            "name": feature_names[i],
            "weight": float(feature_importances[i])
        })

    # --- Save Metrics ---
    metrics = {
        "auroc": auroc,
        "aupr": aupr,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "feature_importances_top25": top_25_importances,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_output_path}")

    # --- Save the trained model ---
    joblib.dump(gbt, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Return the trained model and test set for further use in pipeline.py if needed.
    return gbt, X_test, y_test