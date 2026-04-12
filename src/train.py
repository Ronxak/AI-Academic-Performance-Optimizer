import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
import joblib

def generate_data(n_samples=10000) -> pd.DataFrame:
    np.random.seed(42)
    
    # Feature 1: avg_sleep_7d (4-9)
    avg_sleep_7d = np.random.uniform(4, 9, n_samples)
    
    # Feature 2: sleep_variance (0-2)
    sleep_variance = np.random.uniform(0, 2, n_samples)
    
    # Feature 3: avg_study_7d (0-8)
    avg_study_7d = np.random.uniform(0, 8, n_samples)
    
    # Feature 4: attendance_percent (50-100)
    attendance_percent = np.random.uniform(50, 100, n_samples)
    
    # Feature 5: stress_level (1-10)
    stress_level = np.random.uniform(1, 10, n_samples)
    
    # Feature 6: phone_usage_hours (0-8)
    phone_usage_hours = np.random.uniform(0, 8, n_samples)
    
    # Feature 7: assignment_completion (40-100)
    assignment_completion = np.random.uniform(40, 100, n_samples)
    
    # Feature 8: upcoming_exam (0/1)
    upcoming_exam = np.random.randint(0, 2, n_samples)
    
    # Feature 9: mood_score (1-5)
    mood_score = np.random.uniform(1, 5, n_samples)
    
    # Feature 10: attendance_trend (-5 to +5)
    attendance_trend = np.random.uniform(-5, 5, n_samples)

    # Base target values
    # 1. final_score (0-100)
    score = (
        (avg_study_7d / 8 * 30) + 
        (attendance_percent / 100 * 30) + 
        (assignment_completion / 100 * 30) + 
        (mood_score / 5 * 10) - 
        (stress_level / 10 * 15) - 
        (phone_usage_hours / 8 * 15) - 
        (sleep_variance / 2 * 5)
    )
    # Add noise
    score += np.random.normal(0, 5, n_samples)
    # Clip to realistic bounds
    final_score = np.clip(score + 30, 0, 100)  # +30 just to shift average score up
    
    # 2. burnout_risk (0/1)
    # Higher prob if sleep < 6, study > 5, stress > 7, exam=1
    burnout_prob = (
        (avg_sleep_7d < 6).astype(float) * 0.3 +
        (avg_study_7d > 5).astype(float) * 0.2 +
        (stress_level > 7).astype(float) * 0.3 +
        upcoming_exam * 0.2
    )
    burnout_risk = (np.random.uniform(0, 1, n_samples) < burnout_prob).astype(int)
    
    # 3. absent_next_day (0/1)
    # Depends on sleep, stress, attendance trend, mood
    absent_prob = (
        (avg_sleep_7d < 5).astype(float) * 0.3 +
        (stress_level > 8).astype(float) * 0.3 +
        (attendance_trend < 0).astype(float) * 0.2 +
        (mood_score < 3).astype(float) * 0.2
    )
    absent_next_day = (np.random.uniform(0, 1, n_samples) < absent_prob).astype(int)

    df = pd.DataFrame({
        'avg_sleep_7d': avg_sleep_7d,
        'sleep_variance': sleep_variance,
        'avg_study_7d': avg_study_7d,
        'attendance_percent': attendance_percent,
        'stress_level': stress_level,
        'phone_usage_hours': phone_usage_hours,
        'assignment_completion': assignment_completion,
        'upcoming_exam': upcoming_exam,
        'mood_score': mood_score,
        'attendance_trend': attendance_trend,
        'final_score': final_score,
        'burnout_risk': burnout_risk,
        'absent_next_day': absent_next_day
    })
    
    return df

def main():
    print("Generating synthetic data...")
    df = generate_data()
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/student_data.csv', index=False)
    print("Data saved to data/student_data.csv")

    features = [
        'avg_sleep_7d', 'sleep_variance', 'avg_study_7d', 'attendance_percent',
        'stress_level', 'phone_usage_hours', 'assignment_completion',
        'upcoming_exam', 'mood_score', 'attendance_trend'
    ]

    X = df[features]
    
    # Target 1: absent_next_day -> Attendance Model (predicting attendance probability)
    # Wait, absent_next_day=1 means absent. Attendance prob = 1 - prob(absent)
    y_absent = df['absent_next_day']
    
    # Target 2: final_score -> Performance Model
    y_score = df['final_score']
    
    # Target 3: burnout_risk -> Burnout Model
    y_burnout_risk = df['burnout_risk']

    # Train Test Splits
    X_train, X_test, y_a_train, y_a_test = train_test_split(X, y_absent, test_size=0.2, random_state=42)
    _, _, y_s_train, y_s_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
    _, _, y_b_train, y_b_test = train_test_split(X, y_burnout_risk, test_size=0.2, random_state=42)

    print("\n--- Training Attendance Model ---")
    att_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    att_pipeline.fit(X_train, y_a_train)
    a_preds = att_pipeline.predict(X_test)
    a_probs = att_pipeline.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_a_test, a_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_a_test, a_probs):.4f}")

    print("\n--- Training Performance Model ---")
    perf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])
    perf_pipeline.fit(X_train, y_s_train)
    s_preds = perf_pipeline.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_s_test, s_preds):.4f}")
    print(f"R²: {r2_score(y_s_test, s_preds):.4f}")

    print("\n--- Training Burnout Model ---")
    burn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    burn_pipeline.fit(X_train, y_b_train)
    b_preds = burn_pipeline.predict(X_test)
    b_probs = burn_pipeline.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_b_test, b_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_b_test, b_probs):.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(att_pipeline, 'models/attendance_model.pkl')
    joblib.dump(perf_pipeline, 'models/performance_model.pkl')
    joblib.dump(burn_pipeline, 'models/burnout_model.pkl')
    
    # We also need to save the training data (a sample) for SHAP explainer initialization
    # Because TreeExplainer requires background data when using pipelines sometimes,
    # Or we can just use the X_train sample
    X_train.to_csv('data/X_train_sample.csv', index=False)
    
    print("\nModels saved to models/ directory.")

if __name__ == "__main__":
    main()
