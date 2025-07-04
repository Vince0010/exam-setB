"""
Heart Disease Prediction Model Training Script
This script processes the heart disease dataset, trains a machine learning model,
and saves the trained model and preprocessor for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the dataset"""
    print("Loading dataset...")
    df = pd.read_csv('heart_2020_uncleaned.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nTarget variable distribution:")
    print(df['HeartDisease'].value_counts())
    print(f"Percentage of heart disease cases: {df['HeartDisease'].value_counts()['Yes'] / len(df) * 100:.2f}%")
    
    return df

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    print("\nPreprocessing data...")
    
    # Handle missing values
    # Fill numeric columns with median
    numeric_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val}")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Handle categorical variables
    label_encoders = {}
    
    # Binary categorical variables - simple mapping
    binary_mappings = {
        'HeartDisease': {'No': 0, 'Yes': 1},
        'Smoking': {'no': 0, 'yes': 1},
        'AlcoholDrinking': {'No': 0, 'Yes': 1},
        'Stroke': {'No': 0, 'Yes': 1},
        'DiffWalking': {'No': 0, 'Yes': 1},
        'Sex': {'Female': 0, 'Male': 1},
        'PhysicalActivity': {'No': 0, 'Yes': 1},
        'Asthma': {'No': 0, 'Yes': 1},
        'KidneyDisease': {'No': 0, 'Yes': 1},
        'SkinCancer': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in binary_mappings.items():
        df_processed[col] = df_processed[col].map(mapping)
    
    # Ordinal categorical variables
    ordinal_mappings = {
        'AgeCategory': {
            '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
            '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9,
            '70-74': 10, '75-79': 11, '80 or older': 12
        },
        'GenHealth': {
            'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4
        }
    }
    
    for col, mapping in ordinal_mappings.items():
        df_processed[col] = df_processed[col].map(mapping)
    
    # Handle multi-category variables with Label Encoding
    categorical_cols = ['Race', 'Diabetic']
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Verify no missing values remain
    print(f"Missing values after preprocessing: {df_processed.isnull().sum().sum()}")
    
    return df_processed, label_encoders, binary_mappings, ordinal_mappings

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and select the best one"""
    print("\nTraining models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Select best model based on AUC score
        if auc_score > best_score:
            best_score = auc_score
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")
    return best_model, best_model_name

def save_model_and_preprocessor(model, scaler, encoders, binary_mappings, ordinal_mappings, feature_names):
    """Save the trained model and preprocessing components"""
    print("\nSaving model and preprocessing components...")
    
    # Save model
    with open('heart_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save all preprocessing components
    preprocessing_components = {
        'label_encoders': encoders,
        'binary_mappings': binary_mappings,
        'ordinal_mappings': ordinal_mappings,
        'feature_names': feature_names
    }
    
    with open('preprocessing_components.pkl', 'wb') as f:
        pickle.dump(preprocessing_components, f)
    
    print("Model and preprocessing components saved successfully!")

def main():
    """Main function to run the complete pipeline"""
    print("Heart Disease Prediction Model Training Pipeline")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Preprocess data
    df_processed, label_encoders, binary_mappings, ordinal_mappings = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop('HeartDisease', axis=1)
    y = df_processed['HeartDisease']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    best_model, best_model_name = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save model and preprocessor
    save_model_and_preprocessor(
        best_model, scaler, label_encoders, binary_mappings, 
        ordinal_mappings, list(X.columns)
    )
    
    print("\nTraining pipeline completed successfully!")
    print(f"Best model: {best_model_name}")

if __name__ == "__main__":
    main()
