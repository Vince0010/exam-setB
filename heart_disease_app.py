"""
Heart Disease Prediction Web App
A Streamlit web application for predicting heart disease risk in patients.
Developed for Dr. Mendoza's community health clinic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .risk-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .confidence-score {
        font-size: 1.1rem;
        color: #1976d2;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_components():
    """Load the trained model and preprocessing components"""
    try:
        # Load model
        with open('heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load preprocessing components
        with open('preprocessing_components.pkl', 'rb') as f:
            preprocessing_components = pickle.load(f)
        
        return model, scaler, preprocessing_components
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

def preprocess_input(input_data, scaler, preprocessing_components):
    """Preprocess user input for prediction"""
    try:
        # Get preprocessing components
        binary_mappings = preprocessing_components['binary_mappings']
        ordinal_mappings = preprocessing_components['ordinal_mappings']
        label_encoders = preprocessing_components['label_encoders']
        feature_names = preprocessing_components['feature_names']
        
        # Create a copy of input data
        processed_data = input_data.copy()
        
        # Apply binary mappings
        for col, mapping in binary_mappings.items():
            if col in processed_data and col != 'HeartDisease':
                if processed_data[col] in mapping:
                    processed_data[col] = mapping[processed_data[col]]
        
        # Apply ordinal mappings
        for col, mapping in ordinal_mappings.items():
            if col in processed_data:
                if processed_data[col] in mapping:
                    processed_data[col] = mapping[processed_data[col]]
        
        # Apply label encoders
        for col, encoder in label_encoders.items():
            if col in processed_data:
                try:
                    processed_data[col] = encoder.transform([processed_data[col]])[0]
                except ValueError:
                    # Handle unseen categories
                    processed_data[col] = 0
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([processed_data])
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        return scaled_data
    except Exception as e:
        st.error(f"Error preprocessing input: {str(e)}")
        return None

def main():
    """Main function for the Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Developed for Dr. Mendoza\'s Community Health Clinic</p>', unsafe_allow_html=True)
    
    # Load model and components
    model, scaler, preprocessing_components = load_model_and_components()
    
    if model is None:
        st.error("Please run the training script first to generate the model files.")
        return
    
    # Sidebar for input
    st.sidebar.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
    
    # Collect user input
    input_data = {}
    
    # Basic Information
    st.sidebar.markdown("### Basic Information")
    input_data['BMI'] = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    input_data['Sex'] = st.sidebar.selectbox("Sex", ["Female", "Male"])
    input_data['AgeCategory'] = st.sidebar.selectbox("Age Category", 
        ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
    input_data['Race'] = st.sidebar.selectbox("Race", 
        ["White", "Black", "Asian", "American Indian/Alaskan Native", "Hispanic", "Other"])
    
    # Health Status
    st.sidebar.markdown("### Health Status")
    input_data['GenHealth'] = st.sidebar.selectbox("General Health", 
        ["Poor", "Fair", "Good", "Very good", "Excellent"])
    input_data['PhysicalHealth'] = st.sidebar.number_input("Physical Health (days not good in past 30 days)", 
        min_value=0, max_value=30, value=0)
    input_data['MentalHealth'] = st.sidebar.number_input("Mental Health (days not good in past 30 days)", 
        min_value=0, max_value=30, value=0)
    input_data['SleepTime'] = st.sidebar.number_input("Sleep Time (hours per night)", 
        min_value=1, max_value=24, value=7)
    
    # Medical History
    st.sidebar.markdown("### Medical History")
    input_data['Diabetic'] = st.sidebar.selectbox("Diabetic Status", 
        ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"])
    input_data['Stroke'] = st.sidebar.selectbox("Ever had a stroke?", ["No", "Yes"])
    input_data['Asthma'] = st.sidebar.selectbox("Have asthma?", ["No", "Yes"])
    input_data['KidneyDisease'] = st.sidebar.selectbox("Have kidney disease?", ["No", "Yes"])
    input_data['SkinCancer'] = st.sidebar.selectbox("Have skin cancer?", ["No", "Yes"])
    
    # Lifestyle Factors
    st.sidebar.markdown("### Lifestyle Factors")
    input_data['Smoking'] = st.sidebar.selectbox("Smoking Status", ["no", "yes"])
    input_data['AlcoholDrinking'] = st.sidebar.selectbox("Heavy Alcohol Drinking", ["No", "Yes"])
    input_data['PhysicalActivity'] = st.sidebar.selectbox("Physical Activity in past 30 days", ["No", "Yes"])
    input_data['DiffWalking'] = st.sidebar.selectbox("Difficulty Walking", ["No", "Yes"])
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Patient Summary</h2>', unsafe_allow_html=True)
        
        # Display patient information
        st.write("**Basic Information:**")
        st.write(f"- BMI: {input_data['BMI']}")
        st.write(f"- Sex: {input_data['Sex']}")
        st.write(f"- Age: {input_data['AgeCategory']}")
        st.write(f"- Race: {input_data['Race']}")
        
        st.write("**Health Status:**")
        st.write(f"- General Health: {input_data['GenHealth']}")
        st.write(f"- Physical Health Issues: {input_data['PhysicalHealth']} days")
        st.write(f"- Mental Health Issues: {input_data['MentalHealth']} days")
        st.write(f"- Sleep Time: {input_data['SleepTime']} hours")
        
        st.write("**Medical History:**")
        st.write(f"- Diabetic: {input_data['Diabetic']}")
        st.write(f"- Stroke: {input_data['Stroke']}")
        st.write(f"- Asthma: {input_data['Asthma']}")
        st.write(f"- Kidney Disease: {input_data['KidneyDisease']}")
        st.write(f"- Skin Cancer: {input_data['SkinCancer']}")
        
        st.write("**Lifestyle Factors:**")
        st.write(f"- Smoking: {input_data['Smoking']}")
        st.write(f"- Heavy Drinking: {input_data['AlcoholDrinking']}")
        st.write(f"- Physical Activity: {input_data['PhysicalActivity']}")
        st.write(f"- Difficulty Walking: {input_data['DiffWalking']}")
    
    with col2:
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Predict button
        if st.button("🔍 Predict Heart Disease Risk", type="primary"):
            # Preprocess input
            processed_input = preprocess_input(input_data, scaler, preprocessing_components)
            
            if processed_input is not None:
                # Make prediction
                prediction = model.predict(processed_input)[0]
                prediction_proba = model.predict_proba(processed_input)[0]
                
                # Display results
                confidence = max(prediction_proba) * 100
                risk_probability = prediction_proba[1] * 100
                
                if prediction == 1:
                    st.markdown(f'<div class="prediction-box risk-high">🚨 HIGH RISK of Heart Disease</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box risk-low">✅ LOW RISK of Heart Disease</div>', 
                              unsafe_allow_html=True)
                
                st.markdown(f'<div class="confidence-score">Risk Probability: {risk_probability:.1f}%</div>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-score">Confidence Score: {confidence:.1f}%</div>', 
                          unsafe_allow_html=True)
                
                # Risk factors analysis
                st.markdown("### Risk Factors Analysis")
                risk_factors = []
                
                if input_data['BMI'] > 30:
                    risk_factors.append("High BMI (>30)")
                if input_data['Smoking'] == "yes":
                    risk_factors.append("Smoking")
                if input_data['Diabetic'] in ["Yes", "Yes (during pregnancy)"]:
                    risk_factors.append("Diabetes")
                if input_data['Stroke'] == "Yes":
                    risk_factors.append("Previous stroke")
                if input_data['PhysicalActivity'] == "No":
                    risk_factors.append("No physical activity")
                if input_data['GenHealth'] in ["Poor", "Fair"]:
                    risk_factors.append("Poor general health")
                if input_data['PhysicalHealth'] > 15:
                    risk_factors.append("Frequent physical health issues")
                if input_data['SleepTime'] < 6 or input_data['SleepTime'] > 9:
                    risk_factors.append("Irregular sleep pattern")
                
                if risk_factors:
                    st.write("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.write("**No major risk factors identified.**")
                
                # Recommendations
                st.markdown("### Recommendations")
                recommendations = []
                
                if prediction == 1:
                    recommendations.extend([
                        "Consult with a cardiologist immediately",
                        "Schedule comprehensive cardiac evaluation",
                        "Monitor blood pressure and cholesterol levels",
                        "Consider cardiac stress testing"
                    ])
                
                if input_data['BMI'] > 30:
                    recommendations.append("Consider weight management program")
                if input_data['Smoking'] == "yes":
                    recommendations.append("Smoking cessation program recommended")
                if input_data['PhysicalActivity'] == "No":
                    recommendations.append("Increase physical activity gradually")
                if input_data['SleepTime'] < 6 or input_data['SleepTime'] > 9:
                    recommendations.append("Improve sleep hygiene")
                
                if not recommendations:
                    recommendations.append("Maintain current healthy lifestyle")
                    recommendations.append("Regular check-ups with healthcare provider")
                
                for rec in recommendations:
                    st.write(f"- {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p><strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with healthcare professionals for medical decisions.</p>
        <p>Developed for Dr. Mendoza's Community Health Clinic | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
