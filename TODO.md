# Fix ImportError: cannot import name 'predict_student_status' from 'src.predict'

## Steps:
- [x] Step 1: Edit `src/predict.py` to rename `def predict_student` to `def predict_student_status` (matches imports in main.py and optimize.py).
- [ ] Step 2: Test by running `streamlit run app/main.py` (run `python src/train.py` first if models missing).
- [ ] Step 3: Clean up `src/__innit__.py` → `src/__init__.py` (optional).
- [ ] Done: Import error fixed.
