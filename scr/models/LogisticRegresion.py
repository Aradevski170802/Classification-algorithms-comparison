import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create directories if they don't exist
def create_directories():
    dirs = ['results/metrics', 'results/plots', 'results/models']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# ------------------------
# Utility: Plot Confusion Matrix
# ------------------------
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

# Function to save metrics
def save_metrics(metrics_dict, filename):
    with open(filename, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

# ------------------------
# Preprocessing function that uses pd.get_dummies and returns processed features & target
# We include an argument to optionally align columns using a provided list (for ensuring training and test use same dummy columns)
# ------------------------
def preprocess_data(df, target_column='y', dummy_cols=['job','marital','education','default','housing', 'loan','contact','month','poutcome'], columns_alignment=None):
    # Convert categorical variables to dummies
    df_proc = pd.get_dummies(df, columns=dummy_cols)
    # Convert target variable to binary
    df_proc[target_column] = df_proc[target_column].map({'yes': 1, 'no': 0})
    # Separate features and target
    X = df_proc.drop(target_column, axis=1)
    y = df_proc[target_column]
    
    # If columns_alignment is provided, reindex X to match those columns (fill missing with 0)
    if columns_alignment is not None:
        X = X.reindex(columns=columns_alignment, fill_value=0)
    return X, y

# Create directories at the start
create_directories()

# ------------------------
# Load and preprocess training dataset (bank.csv)
# ------------------------
train_path = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank.csv'
df_train = pd.read_csv(train_path, sep=';')
X_train_full, y_train_full = preprocess_data(df_train)

# Save the training columns to re-use later
training_columns = X_train_full.columns

# Split the training set further into train/val for internal evaluation (optional)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ------------------------
# Train Logistic Regression model on training split
# ------------------------
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Evaluate on the training set
train_preds = lr_model.predict(X_train_scaled)
lr_acc_train = accuracy_score(y_train, train_preds)
lr_report_train = classification_report(y_train, train_preds)
lr_cm_train = confusion_matrix(y_train, train_preds)

print("Training Metrics:")
print("Accuracy: {:.2f}%".format(lr_acc_train * 100))
print("\nClassification Report:\n", lr_report_train)

# Evaluate on the internal validation set
val_preds = lr_model.predict(X_val_scaled)
lr_acc_val = accuracy_score(y_val, val_preds)
lr_report_val = classification_report(y_val, val_preds)
lr_cm_val = confusion_matrix(y_val, val_preds)

print("Validation Metrics:")
print("Accuracy: {:.2f}%".format(lr_acc_val * 100))
print("\nClassification Report:\n", lr_report_val)

# ------------------------
# Now, load and preprocess the full (actual) dataset: bank-full.csv
# ------------------------
full_path = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank-full.csv'
df_full = pd.read_csv(full_path, sep=';')
# Preprocess and align its columns with training set
X_full, y_full = preprocess_data(df_full, columns_alignment=training_columns)

# Scale using the same scaler from training set
X_full_scaled = scaler.transform(X_full)

# Evaluate our Logistic Regression model on the full dataset
full_preds = lr_model.predict(X_full_scaled)
lr_acc_full = accuracy_score(y_full, full_preds)
lr_report_full = classification_report(y_full, full_preds)
lr_cm_full = confusion_matrix(y_full, full_preds)

print("Full Dataset Metrics:")
print("Accuracy: {:.2f}%".format(lr_acc_full * 100))
print("\nClassification Report:\n", lr_report_full)

# Save results with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model
model_path = f'results/models/logistic_regression_{timestamp}.pkl'
joblib.dump(lr_model, model_path)

# Save metrics
metrics = {
    'training': {
        'accuracy': float(lr_acc_train),
        'classification_report': lr_report_train
    },
    'validation': {
        'accuracy': float(lr_acc_val),
        'classification_report': lr_report_val
    },
    'full_dataset': {
        'accuracy': float(lr_acc_full),
        'classification_report': lr_report_full
    }
}
metrics_path = f'results/metrics/logistic_regression_metrics_{timestamp}.json'
save_metrics(metrics, metrics_path)

# Save plots
plot_confusion_matrix(lr_cm_train, 
                     title='Logistic Regression - Training Set',
                     save_path=f'results/plots/confusion_matrix_train_{timestamp}.png')

plot_confusion_matrix(lr_cm_val, 
                     title='Logistic Regression - Validation Set',
                     save_path=f'results/plots/confusion_matrix_val_{timestamp}.png')

plot_confusion_matrix(lr_cm_full, 
                     title='Logistic Regression - Full Dataset',
                     save_path=f'results/plots/confusion_matrix_full_{timestamp}.png')
