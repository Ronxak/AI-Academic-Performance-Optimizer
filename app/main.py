import streamlit as st
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_student_status
from src.optimize import optimize_schedule

st.set_page_config(
    page_title="AI Academic Performance Optimizer",
    page_icon="🎓",
    layout="wide"
)

def main():
    st.title("AI Academic Performance Optimizer")
    st.write("Enter the student's metrics to predict outcomes and get actionable optimization strategies.")

    # Input Section
    st.header("📊 Student Inputs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_sleep_7d = st.slider("Average Sleep (7d) [Hours]", 4.0, 9.0, 7.0, 0.1)
        sleep_variance = st.slider("Sleep Variance", 0.0, 2.0, 0.5, 0.1)
        mood_score = st.slider("Mood Score (1-5)", 1.0, 5.0, 3.0, 0.1)
        
    with col2:
        avg_study_7d = st.slider("Average Study (7d) [Hours]", 0.0, 8.0, 3.0, 0.1)
        stress_level = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, 0.1)
        phone_usage_hours = st.slider("Phone Usage [Hours]", 0.0, 8.0, 3.0, 0.1)
        
    with col3:
        attendance_percent = st.slider("Attendance %", 50.0, 100.0, 85.0, 1.0)
        attendance_trend = st.slider("Attendance Trend", -5.0, 5.0, 0.0, 0.1)
        assignment_completion = st.slider("Assignment Completion %", 40.0, 100.0, 80.0, 1.0)
        upcoming_exam_str = st.selectbox("Upcoming Exam?", ["No", "Yes"])
        upcoming_exam = 1 if upcoming_exam_str == "Yes" else 0

    if st.button("🚀 Analyze", use_container_width=True, type="primary"):
        # Wrap inputs
        input_dict = {
            'avg_sleep_7d': avg_sleep_7d,
            'sleep_variance': sleep_variance,
            'avg_study_7d': avg_study_7d,
            'attendance_percent': attendance_percent,
            'stress_level': stress_level,
            'phone_usage_hours': phone_usage_hours,
            'assignment_completion': assignment_completion,
            'upcoming_exam': upcoming_exam,
            'mood_score': mood_score,
            'attendance_trend': attendance_trend
        }

        with st.spinner("Analyzing data and generating optimizations..."):
            try:
                preds = predict_student_status(input_dict)
                opt = optimize_schedule(input_dict)
            except Exception as e:
                st.error(f"Please ensure models are trained first by running `python src/train.py`. Error: {e}")
                return

        st.divider()

        # CURRENT STATUS
        st.header("Current Status")
        m1, m2, m3 = st.columns(3)
        
        m1.metric("Attendance Probability", f"{preds['attendance_probability']:.1f}%")
        m2.metric("Predicted Score", f"{preds['predicted_score']:.1f} / 100")
        
        # Color code burnout risk
        bp = preds['burnout_probability']
        if bp < 30:
            burnout_color = "🟢 Low"
        elif bp < 70:
            burnout_color = "🟡 Moderate"
        else:
            burnout_color = "🔴 High"
            
        m3.metric("Burnout Risk", f"{bp:.1f}%", delta=burnout_color, delta_color="off")

        st.divider()

        # OPTIMIZATION
        st.header("Optimization Engine")
        st.write("Recommended adjustments to maximize score without increasing burnout.")
        
        o1, o2, o3, o4 = st.columns(4)
        
        sleep_delta = opt['recommended_sleep'] - avg_sleep_7d
        study_delta = opt['recommended_study'] - avg_study_7d
        
        o1.metric("Recommended Sleep", f"{opt['recommended_sleep']:.1f} hr", f"{sleep_delta:+.1f} hr")
        o2.metric("Recommended Study", f"{opt['recommended_study']:.1f} hr", f"{study_delta:+.1f} hr")
        o3.metric("New Predicted Score", f"{opt['new_predicted_score']:.1f}", f"{opt['score_improvement']:+.1f} points")
        
        b_change = opt['burnout_change']
        o4.metric("Burnout Risk Change", f"{opt['new_burnout_probability']:.1f}%", f"{b_change:+.1f}%", delta_color="inverse")

        st.divider()

        # EXPLAINABILITY
        st.header("Key Factors (SHAP)")
        st.write("Top 3 factors affecting the predicted score.")
        
        for factor in preds['top_factors']:
            icon = "⬆️" if factor['effect'] == "increases score" else "⬇️"
            name_clean = factor['name'].replace('_', ' ').title()
            st.info(f"{icon} **{name_clean}**: {factor['effect']} (Impact: {abs(factor['value']):.2f} pts)")

        st.divider()

        # VISUALIZATION
        st.header("Performance Visualization")
        st.write("Feature Importance Analysis")
        
        # Matplotlib Bar Chart
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Prepare data for plotting (top 3)
        plot_factors = preds['top_factors']
        names = [f['name'].replace('_', ' ').title() for f in plot_factors][::-1]
        values = [f['value'] for f in plot_factors][::-1]
        colors = ['#10b981' if v > 0 else '#ef4444' for v in values] # modern UI hex colors
        
        ax.barh(names, values, color=colors)
        ax.set_xlabel("Impact on Score")
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        
        # Remove top & right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
