import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import json
import joblib

def preprocess_data(train_path, test_path):
    # Load the datasets
    train_data = pd.read_csv(train_path, sep=';')
    test_data = pd.read_csv(test_path, sep=';')
    
    # Combine datasets for consistent encoding
    all_data = pd.concat([train_data, test_data], axis=0)
    
    # Encode categorical variables
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                           'contact', 'month', 'poutcome']
    encoder = LabelEncoder()
    for col in categorical_columns:
        all_data[col] = encoder.fit_transform(all_data[col])
    
    # Encode target variable
    all_data['y'] = all_data['y'].map({'yes': 1, 'no': 0})
    
    # Split back into train and test
    train_data = all_data[:len(train_data)]
    test_data = all_data[len(train_data):]
    
    # Split features and target
    X_train = train_data.drop(columns=['y'])
    y_train = train_data['y']
    X_test = test_data.drop(columns=['y'])
    y_test = test_data['y']
    
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    
    return metrics

def visualize_tree(model, feature_names):
    tree_rules = export_text(model, feature_names=feature_names)
    print(tree_rules)
    
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=['No', 'Yes'], filled=True)
    plt.show()

def save_results(metrics, model):
    with open('results/metrics/decision_tree_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    joblib.dump(model, 'results/models/decision_tree.pkl')

if __name__ == "__main__":
    train_path = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank.csv'
    test_path = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank-full.csv'
    
    X_train, X_test, y_train, y_test = preprocess_data(train_path, test_path)
    
    dt_model = train_decision_tree(X_train, y_train)
    
    metrics = evaluate_model(dt_model, X_test, y_test)
    
    visualize_tree(dt_model, feature_names=X_train.columns)
    
    save_results(metrics, dt_model)
