import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import json
import joblib
from imblearn.over_sampling import SMOTE  # Import SMOTE

def preprocess_data(train_path, test_path):
    # Load datasets using ';' as the delimiter
    train_data = pd.read_csv(train_path, sep=';')
    test_data = pd.read_csv(test_path, sep=';')
    
    # Combine both datasets to ensure consistent encoding
    all_data = pd.concat([train_data, test_data], axis=0)
    
    # List of categorical columns to encode
    categorical_columns = ['job', 'marital', 'education', 'default',
                           'housing', 'loan', 'contact', 'month', 'poutcome']
    encoder = LabelEncoder()
    for col in categorical_columns:
        all_data[col] = encoder.fit_transform(all_data[col])
    
    # Map target variable from 'yes'/'no' to 1/0
    all_data['y'] = all_data['y'].map({'yes': 1, 'no': 0})
    
    # Split the combined data back into training and test sets
    n_train = train_data.shape[0]
    train_data = all_data.iloc[:n_train, :]
    test_data = all_data.iloc[n_train:, :]
    
    X_train = train_data.drop(columns=['y'])
    y_train = train_data['y']
    X_test = test_data.drop(columns=['y'])
    y_test = test_data['y']
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Update these paths to your local files
    train_path = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank.csv'
    test_path  = r'C:\Users\frogo\Desktop\Classification-algorithms-comparison\dataset\archive\bank-full.csv'
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(train_path, test_path)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    
    # Scale the features since SVMs are sensitive to feature scales
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Apply SMOTE on the scaled training data to deal with class imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print("After SMOTE, training data shape:", X_train_smote.shape)
    
    # Create and train the SVM model (here using the RBF kernel)
    svm_model = SVC(kernel='rbf', random_state=42, C=1.0)
    svm_model.fit(X_train_smote, y_train_smote)
    
    # Make predictions on the test set
    y_pred = svm_model.predict(X_test_scaled)
    
    # Evaluate the model
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("\nAccuracy:", accuracy)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("SVM Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.colorbar()
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.show()
    
    # Compute ROC curve and AUC score using decision_function
    y_scores = svm_model.decision_function(X_test_scaled)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], 'k--')  # Diagonal line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # Save the scaler, model, and evaluation metrics
    joblib.dump(scaler, 'results/models/svm_scaler.pkl')
    joblib.dump(svm_model, 'results/models/svm_model.pkl')
    
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": roc_auc
    }
    with open('results/metrics/svm_metrics.json', 'w') as f:
        json.dump(metrics, f)
