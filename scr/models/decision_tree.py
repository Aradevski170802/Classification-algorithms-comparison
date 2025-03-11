import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_curve, auc)
import matplotlib.pyplot as plt
import json
import joblib
from imblearn.over_sampling import SMOTE  # Optional: for handling imbalanced data

# -------------------- Data Preprocessing --------------------

def preprocess_data(train_path, test_path):
    """
    Loads training and test datasets, performs label encoding on categorical features,
    maps the target variable (y) from yes/no to 1/0, and splits the combined data back 
    into training and test sets.
    """
    # Load datasets (delimiter is ';')
    train_data = pd.read_csv(train_path, sep=';')
    test_data = pd.read_csv(test_path, sep=';')
    
    # Combine datasets to ensure consistent encoding
    all_data = pd.concat([train_data, test_data], axis=0)
    
    # List of categorical columns
    categorical_columns = ['job', 'marital', 'education', 'default', 
                           'housing', 'loan', 'contact', 'month', 'poutcome']
    encoder = LabelEncoder()
    for col in categorical_columns:
        all_data[col] = encoder.fit_transform(all_data[col])
    
    # Map target variable from 'yes'/'no' to 1/0
    all_data['y'] = all_data['y'].map({'yes': 1, 'no': 0})
    
    # Split data back based on the original training set length
    n_train = train_data.shape[0]
    train_data = all_data.iloc[:n_train, :]
    test_data = all_data.iloc[n_train:, :]
    
    # Separate features and target
    X_train = train_data.drop(columns=['y'])
    y_train = train_data['y']
    X_test = test_data.drop(columns=['y'])
    y_test = test_data['y']
    
    return X_train, X_test, y_train, y_test

# -------------------- (Optional) Feature Scaling --------------------
# Although decision trees do not require feature scaling, we include it
# for consistency with other parts of the project.

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# -------------------- Model Training with Hyperparameter Tuning --------------------

def tune_decision_tree(X_train, y_train, use_smote=False):
    """
    Optionally applies SMOTE to balance the training set,
    then tunes a DecisionTreeClassifier using GridSearchCV.
    Returns the best estimator found.
    """
    # Optionally balance the training data if there is class imbalance
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("After SMOTE, training data shape:", X_train.shape)
    
    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    # Initialize the decision tree classifier
    dt = DecisionTreeClassifier(random_state=42)
    
    # Set up GridSearchCV (using 5-fold cross-validation)
    grid_search = GridSearchCV(dt, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

# -------------------- Evaluation and Visualization --------------------

def evaluate_model(model, X_test, y_test):
    """
    Predicts on the test set, prints and returns evaluation metrics,
    and plots the ROC curve.
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("\nAccuracy:", acc)
    
    # ROC Curve and AUC
    # Decision trees provide predict_proba method
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree')
    plt.legend(loc="lower right")
    plt.show()
    
    # Optionally, plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Decision Tree Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.colorbar()
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.show()
    
    # Return metrics as a dictionary
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": roc_auc
    }
    return metrics

def visualize_tree(model, feature_names):
    """
    Prints a text representation of the decision tree and plots its graphical structure.
    """
    tree_rules = export_text(model, feature_names=list(feature_names))
    print("Decision Tree Rules:")
    print(tree_rules)
    
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=['No', 'Yes'], filled=True)
    plt.title("Decision Tree Structure")
    plt.show()

def save_results(metrics, model, scaler=None):
    """
    Saves evaluation metrics to a JSON file, the trained model, and optionally the scaler.
    """
    # Save metrics as JSON
    with open('results/metrics/decision_tree_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save the trained model
    joblib.dump(model, 'results/models/decision_tree_model.pkl')
    
    # Optionally save scaler
    if scaler is not None:
        joblib.dump(scaler, 'results/models/decision_tree_scaler.pkl')

# -------------------- Main Workflow --------------------

if __name__ == "__main__":
    # Update these paths to your local dataset files
    train_path = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank.csv'
    test_path  = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank-full.csv'
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(train_path, test_path)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    
    # (Optional) Scale features. Note: Decision trees are scale invariant, so scaling is not mandatory.
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Choose whether to train on scaled data. Decision trees work fine without scaling.
    # For GridSearchCV and consistency, we'll use the (unscaled) X_train here.
    # If you want to use scaled data, use X_train_scaled and X_test_scaled instead.
    use_scaled = False
    
    if use_scaled:
        dt_input_train, dt_input_test = X_train_scaled, X_test_scaled
    else:
        dt_input_train, dt_input_test = X_train, X_test
    
    # (Optional) If your data is imbalanced, you can apply SMOTE on the training set.
    use_smote = False  # Set to True if desired
    if use_smote:
        smote = SMOTE(random_state=42)
        dt_input_train, y_train = smote.fit_resample(dt_input_train, y_train)
        print("After SMOTE, training shape:", dt_input_train.shape)
    
    # Hyperparameter tuning using GridSearchCV to find the best decision tree model
    best_dt_model = tune_decision_tree(dt_input_train, y_train, use_smote=False)
    
    # Evaluate the best model on the test set
    metrics = evaluate_model(best_dt_model, dt_input_test, y_test)
    
    # Visualize the decision tree
    visualize_tree(best_dt_model, X_train.columns)
    
    # Save the evaluation metrics, model, and scaler (if used)
    save_results(metrics, best_dt_model, scaler if use_scaled else None)
