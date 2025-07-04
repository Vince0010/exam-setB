# Heart Disease Risk Prediction System

A comprehensive machine learning solution for predicting heart disease risk in patients, designed specifically for Dr. Mendoza's Community Health Clinic.

## 🎯 Project Overview

This system provides healthcare professionals with a user-friendly tool to assess patient heart disease risk using various health metrics. The application features three risk categories (Low, Medium, High) with confidence scores to support clinical decision-making.

## 🔧 Features

### 🤖 Machine Learning Models
- **Random Forest Classifier**: Robust ensemble method for accurate predictions
- **Logistic Regression**: Interpretable linear model for baseline comparison
- **Ensemble Prediction**: Combines both models for improved accuracy

### 📊 Risk Assessment
- **Three-tier Risk Classification**: Low, Medium, and High risk categories
- **Confidence Scores**: Probability-based confidence metrics
- **Model Comparison**: Side-by-side comparison of different algorithms

### 🌐 Web Application
- **Streamlit Interface**: Clean, professional healthcare-appropriate design
- **Real-time Predictions**: Instant risk assessment
- **Personalized Recommendations**: Tailored health advice based on risk factors
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required dataset: `heart_2020_uncleaned.csv`

### Installation & Setup

1. **Clone or download** the project files to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**:
   ```bash
   python heart_disease_model.py
   ```

4. **Test the models** (optional):
   ```bash
   python test_model.py
   ```

5. **Launch the web application**:
   ```bash
   streamlit run heart_disease_app.py
   ```

### Automated Deployment
For one-click deployment, run:
```bash
python deploy.py
```

## 📋 Input Parameters

The system accepts the following patient information:

### Basic Health Metrics
- **BMI**: Body Mass Index (10.0 - 60.0)
- **Physical Health**: Days of poor physical health in past month (0-30)
- **Mental Health**: Days of poor mental health in past month (0-30)
- **Sleep Time**: Average hours of sleep per night (2-24)

### Medical History
- **Smoking Status**: Current smoking status (Yes/No)
- **Alcohol Consumption**: Heavy drinking history (Yes/No)
- **Stroke History**: Previous stroke (Yes/No)
- **Walking Difficulty**: Difficulty walking or climbing stairs (Yes/No)
- **Diabetes**: Diabetic status (No/Yes/Borderline/Gestational)
- **Asthma**: Asthma diagnosis (Yes/No)
- **Kidney Disease**: Kidney disease diagnosis (Yes/No)
- **Skin Cancer**: Skin cancer diagnosis (Yes/No)

### Demographics
- **Sex**: Male/Female
- **Age Category**: Age ranges from 18-24 to 80+
- **Race**: Ethnicity classification
- **Physical Activity**: Recent physical activity (Yes/No)
- **General Health**: Self-reported health status

## 🎯 Risk Categories

### 🟢 Low Risk (< 30% probability)
- Minimal heart disease risk
- Maintain current healthy habits
- Regular preventive care recommended

### 🟡 Medium Risk (30-70% probability)
- Moderate heart disease risk
- Lifestyle modifications recommended
- Increased monitoring suggested

### 🔴 High Risk (> 70% probability)
- Significant heart disease risk
- Immediate medical consultation recommended
- Comprehensive risk factor management needed

## 📊 Model Performance

- **Random Forest Accuracy**: ~91.6%
- **Logistic Regression Accuracy**: ~87.3%
- **Ensemble AUC Score**: ~0.84
- **Dataset Size**: 319,795 patient records

## 🔍 File Structure

```
exam-setB/
├── heart_disease_model.py      # Model training and preprocessing
├── heart_disease_app.py        # Streamlit web application
├── test_model.py              # Model testing and validation
├── deploy.py                  # Automated deployment script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── heart_2020_uncleaned.csv   # Dataset (user-provided)
└── Generated Files:
    ├── rf_model.pkl           # Trained Random Forest model
    ├── lr_model.pkl           # Trained Logistic Regression model
    ├── scaler.pkl             # Feature scaling object
    ├── label_encoders.pkl     # Categorical encoders
    ├── target_encoder.pkl     # Target variable encoder
    └── feature_names.pkl      # Feature names for consistency
```

## 🛠️ Troubleshooting

### Common Issues

1. **Dataset not found**
   - Ensure `heart_2020_uncleaned.csv` is in the project directory
   - Check file name spelling and case sensitivity

2. **Model files missing**
   - Run `python heart_disease_model.py` to train models
   - Ensure all .pkl files are generated

3. **Package installation errors**
   - Update pip: `pip install --upgrade pip`
   - Use virtual environment if needed

4. **Streamlit won't start**
   - Check if port 8501 is available
   - Try: `streamlit run heart_disease_app.py --server.port 8502`

## ⚠️ Important Notes

### Medical Disclaimer
This prediction tool is designed for educational and screening purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

### Data Privacy
- No patient data is stored or transmitted
- All processing occurs locally
- Complies with healthcare privacy standards

---

**Developed for Dr. Mendoza's Community Health Clinic**  
*Improving patient care through predictive analytics*