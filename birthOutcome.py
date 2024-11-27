import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
from sklearn import metrics

# Hidden author comment: Code refactored and generalized by [Your Name].

def load_dataset(filepath):
    """Load dataset from a file."""
    dataset = pd.read_csv(filepath)
    print(f"Dataset shape: {dataset.shape}")
    print(dataset.isnull().sum())
    return dataset

def impute_missing_values(dataset, mean_impute_cols, locf_impute_cols):
    """Impute missing values using mean and LOCF methods."""
    for col in mean_impute_cols:
        dataset[col] = dataset[col].fillna(dataset[col].mean()).round(0)
    for col in locf_impute_cols:
        dataset[col] = dataset[col].fillna(method='ffill')
    print("After Imputation:")
    print(dataset.isnull().sum())
    return dataset

def preprocess_data(dataset, target_col, test_size=0.3, random_state=7):
    """Split the data into train and test sets and apply scaling."""
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test, X.columns

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, cv=5):
    """Train a model and evaluate its performance with cross-validation."""
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"Accuracy: {acc}")
    print(f"Cross-validation scores: {scores}")
    print(f"Average CV Score: {np.mean(scores)}")
    return model

def plot_roc_curves(models, X_test, y_test, model_names):
    """Plot ROC curves for given models."""
    plt.figure(figsize=(10, 5))
    lw = 2
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    for model, name in zip(models, model_names):
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f"{name} ROC (area = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.show()

def feature_importance_plot(model, feature_names, title="Feature Importance"):
    """Plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Main Script
if __name__ == "__main__":
    # Configurable file paths and column names
    input_filepath = "data/input_data.csv"
    output_filepath = "data/output_data.csv"
    mean_impute_cols = ["Column1", "Column2", "Column3"]  # Replace with actual column names
    locf_impute_cols = ["Column4", "Column5", "Column6"]  # Replace with actual column names
    target_col = "Target"  # Replace with actual target column name

    # Load and preprocess the data
    dataset = load_dataset(input_filepath)
    dataset = impute_missing_values(dataset, mean_impute_cols, locf_impute_cols)

    # Save and reload the dataset
    dataset.to_csv(output_filepath, index=False)
    dataset = pd.read_csv(output_filepath)

    # Preprocess and split data
    X_train_std, X_test_std, y_train, y_test, feature_names = preprocess_data(dataset, target_col)

    # Initialize models
    models = [
        LogisticRegression(solver="liblinear", multi_class="ovr", random_state=1),
        SVC(gamma="auto", probability=True, random_state=1),
        RandomForestClassifier(n_estimators=60, random_state=1),
    ]
    model_names = ["Logistic Regression", "SVM", "Random Forest"]

    # Train and evaluate each model
    trained_models = [
        train_and_evaluate_model(model, X_train_std, X_test_std, y_train, y_test)
        for model in models
    ]

    # Plot ROC curves
    plot_roc_curves(trained_models, X_test_std, y_test, model_names)

    # Feature Importance for Random Forest
    feature_importance_plot(trained_models[2], feature_names, title="Feature Importance in Data")
