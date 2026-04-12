# 🎓 AI Academic Performance & Wellbeing Optimizer

A production-grade AI system designed to predict student outcomes and provide actionable optimization strategies using Machine Learning and SHAP explainability.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## ✨ Features

- **Multi-Output Predictions**: Simultaneously predicts Attendance Probability, Academic Score, and Burnout Risk.
- **Optimization Engine**: An exhaustive search algorithm that recommends specific adjustments (Study/Sleep) to maximize scores while maintaining wellbeing.
- **SHAP Explainability**: Visualizes the top factors influencing each prediction for transparency.
- **Modern Dashboard**: A clean, responsive Streamlit interface with interactive visualizations.

## 📊 Model Performance

The current models are trained on a high-fidelity synthetic dataset (10,000 samples) with the following verified metrics:

| Model | Primary Metric | Score |
| :--- | :--- | :--- |
| **Performance (Regressor)** | R² Score | **0.8114** |
| **Attendance (Classifier)** | ROC-AUC | **0.7679** |
| **Wellbeing (Classifier)** | ROC-AUC | **0.7856** |

## 🚀 Deployment (Streamlit Cloud)

1. **Push to GitHub**: Make sure this repository is pushed to your GitHub.
2. **Authorize Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. **Deploy**:
   - Repository: `your-username/student_ai_system`
   - Main file path: `app/main.py`
4. **Done!** Your AI system will be live at a public URL.

## 🛠 Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models (Optional)**:
   ```bash
   python src/train.py
   ```

3. **Launch Dashboard**:
   ```bash
   streamlit run app/main.py
   ```

---
*Created with ❤️ for Academic Excellence.*
