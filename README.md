# AI Academic Performance & Wellbeing Optimization System

A complete end-to-end machine learning web application built with Streamlit and scikit-learn.

## Project Structure
- `data/`: Contains generated datasets.
- `models/`: Stores trained models (.pkl).
- `src/`: Core Python modules for ML logic (training, prediction, optimization).
- `app/`: Streamlit web interface.

## Quickstart

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate data and train models:
   ```bash
   python src/train.py
   ```

3. Run the dashboard:
   ```bash
   streamlit run app/main.py
   ```
