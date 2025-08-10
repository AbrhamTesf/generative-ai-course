import os
import time
import joblib
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

import mlflow
import mlflow.sklearn

def train_model():
    """
    Loads data, trains a RandomForestClassifier, and saves the model, metrics, and confusion matrix plot.
    Logs all information to MLflow.
    """
    print("Starting model training...")

    # Define paths
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '..', 'data', 'processed')
    models_dir = os.path.join(script_dir, '..', 'models')
    config_path = os.path.join(script_dir, '..', 'config', 'train_config.yaml')
    os.makedirs(models_dir, exist_ok=True)

    # Load config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")

    # Start an MLflow run
    mlflow.set_experiment("Fraud Detection RandomForest Training")
    with mlflow.start_run():
        start_time = time.time()
        
        # Log hyperparameters
        mlflow.log_params(config['model'])

        # Load preprocessed data
        X_train = joblib.load(os.path.join(data_dir, 'X_train.joblib'))
        y_train = joblib.load(os.path.join(data_dir, 'y_train.joblib'))
        X_test = joblib.load(os.path.join(data_dir, 'X_test.joblib'))
        y_test = joblib.load(os.path.join(data_dir, 'y_test.joblib'))
        print("Data loaded successfully.")

        # Initialize and train the RandomForestClassifier
        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(**config['model'])
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_f1_score": f1,
            "test_precision": precision,
            "test_recall": recall
        })

        print("\n--- Model Evaluation on Test Set ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Save the trained model locally (optional for MLflow)
        model_path = os.path.join(models_dir, 'model.pkl')
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Log the trained model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model logged to MLflow.")
        
        # Calculate and print cross-validation scores
        print("\n--- Cross-Validation Scores (5-fold) ---")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"F1-Scores: {cv_scores}")
        print(f"Mean F1-Score: {cv_scores.mean():.4f}")
        mlflow.log_metric("cv_mean_f1_score", cv_scores.mean())

        # Log training runtime
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Training completed in {runtime:.2f} seconds.")
        mlflow.log_metric("training_runtime_sec", runtime)

        # Save metrics to a JSON file (optional for MLflow, but good for local use)
        metrics = {
            'test_accuracy': accuracy,
            'test_f1_score': f1,
            'test_precision': precision,
            'test_recall': recall,
            'cv_mean_f1_score': cv_scores.mean(),
            'training_runtime_sec': runtime,
            'n_estimators': config['model']['n_estimators'],
            'max_depth': config['model']['max_depth']
        }
        metrics_path = os.path.join(os.path.dirname(__file__), '..', 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved to metrics.json")

        # Plot and save confusion matrix
        print("Generating confusion matrix plot...")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraudulent"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        
        plot_path = os.path.join(models_dir, 'confusion_matrix.png')
        plt.savefig(plot_path)
        print(f"Confusion matrix plot saved to {plot_path}")
        plt.close()
        
        # Log the plot as an artifact
        mlflow.log_artifact(plot_path, "confusion_matrix_plot")
        print("Confusion matrix plot logged to MLflow as an artifact.")


if __name__ == "__main__":
    train_model()