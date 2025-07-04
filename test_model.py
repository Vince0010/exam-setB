"""
Test script to verify the heart disease prediction model is working correctly.
"""

import pandas as pd
import numpy as np
import pickle

def test_model():
    """Test the trained model with sample data"""
    print("Testing Heart Disease Prediction Model")
    print("=" * 40)
    
    # Load model and components
    try:
        with open('heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('preprocessing_components.pkl', 'rb') as f:
            preprocessing_components = pickle.load(f)
        
        print("✓ Model and components loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Test case 1: High risk patient
    print("\n--- Test Case 1: High Risk Patient ---")
    high_risk_patient = {
        'BMI': 35.0,
        'Smoking': 1,  # Yes
        'AlcoholDrinking': 1,  # Yes
        'Stroke': 1,  # Yes
        'PhysicalHealth': 20.0,
        'MentalHealth': 15.0,
        'DiffWalking': 1,  # Yes
        'Sex': 1,  # Male
        'AgeCategory': 11,  # 75-79
        'Race': 0,  # Will be encoded
        'Diabetic': 1,  # Yes
        'PhysicalActivity': 0,  # No
        'GenHealth': 0,  # Poor
        'SleepTime': 4.0,
        'Asthma': 1,  # Yes
        'KidneyDisease': 1,  # Yes
        'SkinCancer': 1  # Yes
    }
    
    # Test case 2: Low risk patient
    print("\n--- Test Case 2: Low Risk Patient ---")
    low_risk_patient = {
        'BMI': 22.0,
        'Smoking': 0,  # No
        'AlcoholDrinking': 0,  # No
        'Stroke': 0,  # No
        'PhysicalHealth': 0.0,
        'MentalHealth': 0.0,
        'DiffWalking': 0,  # No
        'Sex': 0,  # Female
        'AgeCategory': 2,  # 30-34
        'Race': 0,  # Will be encoded
        'Diabetic': 0,  # No
        'PhysicalActivity': 1,  # Yes
        'GenHealth': 4,  # Excellent
        'SleepTime': 8.0,
        'Asthma': 0,  # No
        'KidneyDisease': 0,  # No
        'SkinCancer': 0  # No
    }
    
    # Test both cases
    test_cases = [
        ("High Risk Patient", high_risk_patient),
        ("Low Risk Patient", low_risk_patient)
    ]
    
    for case_name, patient_data in test_cases:
        print(f"\n{case_name}:")
        
        # Create DataFrame
        df = pd.DataFrame([patient_data])
        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        prediction_proba = model.predict_proba(scaled_data)[0]
        
        risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
        confidence = max(prediction_proba) * 100
        risk_probability = prediction_proba[1] * 100
        
        print(f"  Prediction: {risk_level}")
        print(f"  Risk Probability: {risk_probability:.1f}%")
        print(f"  Confidence: {confidence:.1f}%")
    
    print("\n✓ Model testing completed successfully!")

if __name__ == "__main__":
    test_model()
