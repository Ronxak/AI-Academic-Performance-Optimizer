import os
import joblib
import pandas as pd
import numpy as np
import shap

# Global initialization to avoid reloading models on every request
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

_MODELS_CACHE = {}

def load_models():
    """Load models lazily and cache them in memory."""
    global _MODELS_CACHE
    if not _MODELS_CACHE:
        attendance_model = joblib.load(os.path.join(MODELS_DIR, 'attendance_model.pkl'))
        performance_model = joblib.load(os.path.join(MODELS_DIR, 'performance_model.pkl'))
        burnout_model = joblib.load(os.path.join(MODELS_DIR, 'burnout_model.pkl'))
        
        # Load background data for SHAP
        bg_data_path = os.path.join(DATA_DIR, 'X_train_sample.csv')
        if os.path.exists(bg_data_path):
            bg_data = pd.read_csv(bg_data_path)
        else:
            bg_data = None
            
        _MODELS_CACHE = (attendance_model, performance_model, burnout_model, bg_data)
        
    return _MODELS_CACHE

def predict_student_status(input_dict: dict) -> dict:
    attendance_model, performance_model, burnout_model, bg_data = load_models()
    
    # Convert input to DataFrame
    # Need to make sure feature order is consistent
    features = [
        'avg_sleep_7d', 'sleep_variance', 'avg_study_7d', 'attendance_percent',
        'stress_level', 'phone_usage_hours', 'assignment_completion',
        'upcoming_exam', 'mood_score', 'attendance_trend'
    ]
    df = pd.DataFrame([input_dict], columns=features)
    
    # 1. Predict Attendance (Wait, model predicts probability of absence)
    prob_absent = attendance_model.predict_proba(df)[0, 1]
    attendance_probability = (1.0 - prob_absent) * 100.0  # Convert to percentage
    
    # 2. Predict Performance Score
    predicted_score = performance_model.predict(df)[0]
    
    # 3. Predict Burnout Risk
    burnout_probability = burnout_model.predict_proba(df)[0, 1] * 100.0
    
    # 4. Extract SHAP values for Performance Model
    # Since we used a pipeline, we need to scale the data first to pass it to TreeExplainer
    scaler = performance_model.named_steps['scaler']
    rf_model = performance_model.named_steps['rf']
    
    scaled_df = scaler.transform(df)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(scaled_df)[0]
    
    # Get top 3 factors
    feature_importance = pd.DataFrame({
        'feature': features,
        'shap_value': shap_vals
    })
    
    # Sort by absolute magnitude of SHAP value
    feature_importance['abs_shap_value'] = feature_importance['shap_value'].abs()
    top_factors_df = feature_importance.sort_values(by='abs_shap_value', ascending=False).head(3)
    
    top_factors = []
    for _, row in top_factors_df.iterrows():
        top_factors.append({
            'name': row['feature'],
            'effect': 'increases score' if row['shap_value'] > 0 else 'decreases score',
            'value': row['shap_value']
        })

    # Return full dictionary
    return {
        'attendance_probability': attendance_probability,
        'predicted_score': predicted_score,
        'burnout_probability': burnout_probability,
        'top_factors': top_factors,
        'shap_values_raw': shap_vals.tolist(),
        'base_value': explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
        'features': features
    }
