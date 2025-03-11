import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc)
from imblearn.over_sampling import SMOTE

# ---------------------------
# Create directories if they don't exist
# ---------------------------
def create_directories():
    dirs = ['results/metrics', 'results/plots', 'results/models']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

create_directories()

# ---------------------------
# Utility: Plot Confusion Matrix
# ---------------------------
def plot_confusion_matrix(cm, title='Confusion Matrix', save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Subscribed', 'Subscribed'],
                yticklabels=['Not Subscribed', 'Subscribed'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ---------------------------
# Utility: Plot ROC Curve
# ---------------------------
def plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve', save_path=None):
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ---------------------------
# Preprocessing Function using get_dummies
# ---------------------------
def preprocess_data(df, target_column='y', dummy_cols=['job','marital','education','default',
                                                       'housing', 'loan','contact','month','poutcome'],
                    columns_alignment=None):
    # One-hot encode categorical variables
    df_proc = pd.get_dummies(df, columns=dummy_cols)
    # Convert target variable to binary
    df_proc[target_column] = df_proc[target_column].map({'yes': 1, 'no': 0})
    # Separate features and target
    X = df_proc.drop(target_column, axis=1)
    y = df_proc[target_column]
    # If alignment is needed (for full dataset), reindex to training columns
    if columns_alignment is not None:
        X = X.reindex(columns=columns_alignment, fill_value=0)
    return X, y

# ---------------------------
# File paths
# ---------------------------
train_path = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank.csv'
full_path  = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank-full.csv'

# ---------------------------
# Load and preprocess training data
# ---------------------------
df_train = pd.read_csv(train_path, sep=';')
X_train_full, y_train_full = preprocess_data(df_train)
# Save training columns for full dataset alignment
training_columns = X_train_full.columns

# Split training into train/validation for internal evaluation
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Scale features (Logistic Regression is sensitive to feature scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ---------------------------
# (Optional) Balance the training data using SMOTE
# ---------------------------
use_smote = False  # Set True if data imbalance is an issue
if use_smote:
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    print("After SMOTE, X_train shape:", X_train_scaled.shape)

# ---------------------------
# Hyperparameter Tuning with GridSearchCV for Logistic Regression
# ---------------------------
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # 'liblinear' supports L1 penalty
}
lr = LogisticRegression(random_state=42, max_iter=1000)
grid_search = GridSearchCV(lr, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_lr_model = grid_search.best_estimator_

# ---------------------------
# Evaluate on internal validation set
# ---------------------------
val_preds = best_lr_model.predict(X_val_scaled)
lr_acc_val = accuracy_score(y_val, val_preds)
lr_report_val = classification_report(y_val, val_preds)
lr_cm_val = confusion_matrix(y_val, val_preds)

print("Validation Metrics:")
print("Accuracy: {:.2f}%".format(lr_acc_val * 100))
print("\nClassification Report:\n", lr_report_val)

# Compute ROC and AUC on validation set
# For Logistic Regression: use predict_proba to get probabilities
val_probs = best_lr_model.predict_proba(X_val_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_val, val_probs)
roc_auc_val = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, roc_auc_val, title='Logistic Regression ROC - Validation Set',
               save_path=f'results/plots/roc_curve_val.png')

plot_confusion_matrix(lr_cm_val, title='Logistic Regression - Validation Confusion Matrix',
                      save_path=f'results/plots/confusion_matrix_val.png')

# ---------------------------
# Load and preprocess full dataset
# ---------------------------
df_full = pd.read_csv(full_path, sep=';')
X_full, y_full = preprocess_data(df_full, columns_alignment=training_columns)
X_full_scaled = scaler.transform(X_full)

# Evaluate on the full dataset
full_preds = best_lr_model.predict(X_full_scaled)
lr_acc_full = accuracy_score(y_full, full_preds)
lr_report_full = classification_report(y_full, full_preds)
lr_cm_full = confusion_matrix(y_full, full_preds)

print("Full Dataset Metrics:")
print("Accuracy: {:.2f}%".format(lr_acc_full * 100))
print("\nClassification Report:\n", lr_report_full)

full_probs = best_lr_model.predict_proba(X_full_scaled)[:, 1]
fpr_full, tpr_full, _ = roc_curve(y_full, full_probs)
roc_auc_full = auc(fpr_full, tpr_full)
plot_roc_curve(fpr_full, tpr_full, roc_auc_full, title='Logistic Regression ROC - Full Dataset',
               save_path=f'results/plots/roc_curve_full.png')

plot_confusion_matrix(lr_cm_full, title='Logistic Regression - Full Dataset Confusion Matrix',
                      save_path=f'results/plots/confusion_matrix_full.png')

# ---------------------------
# Save the model, scaler, and metrics with timestamp
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'results/models/logistic_regression_{timestamp}.pkl'
joblib.dump(best_lr_model, model_path)
scaler_path = f'results/models/logistic_regression_scaler_{timestamp}.pkl'
joblib.dump(scaler, scaler_path)

metrics = {
    'validation': {
        'accuracy': float(lr_acc_val),
        'classification_report': lr_report_val,
        'confusion_matrix': lr_cm_val.tolist(),
        'roc_auc': roc_auc_val
    },
    'full_dataset': {
        'accuracy': float(lr_acc_full),
        'classification_report': lr_report_full,
        'confusion_matrix': lr_cm_full.tolist(),
        'roc_auc': roc_auc_full
    }
}
metrics_path = f'results/metrics/logistic_regression_metrics_{timestamp}.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
