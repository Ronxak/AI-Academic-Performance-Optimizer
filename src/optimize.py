import numpy as np
import copy
from src.predict import predict_student_status

def optimize_schedule(input_dict: dict) -> dict:
    """
    Finds the optimal sleep and study adjustments.
    Try increasing sleep and study by 0.5 to 2 hours.
    Constraints: sleep <= 9, study <= 10
    Constraint: new_burnout_probability <= current_burnout_probability
    Maximize: predicted_score
    """
    # Get baseline predictions
    current_preds = predict_student_status(input_dict)
    current_score = current_preds['predicted_score']
    current_burnout = current_preds['burnout_probability']
    current_sleep = input_dict['avg_sleep_7d']
    current_study = input_dict['avg_study_7d']

    best_plan = None
    best_score = current_score
    best_burnout = current_burnout
    best_sleep = current_sleep
    best_study = current_study

    # Exhaustive search over increments of 0.5 up to 2.0
    increments = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    for sleep_inc in increments:
        for study_inc in increments:
            if sleep_inc == 0 and study_inc == 0:
                continue
                
            test_sleep = current_sleep + sleep_inc
            test_study = current_study + study_inc
            
            if test_sleep > 9.0 or test_study > 10.0:
                continue
                
            # Create a test candidate
            test_dict = copy.deepcopy(input_dict)
            test_dict['avg_sleep_7d'] = test_sleep
            test_dict['avg_study_7d'] = test_study
            
            candidate_preds = predict_student_status(test_dict)
            candidate_score = candidate_preds['predicted_score']
            candidate_burnout = candidate_preds['burnout_probability']
            
            # Constraint check
            if candidate_burnout <= current_burnout:
                # Is it better?
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_burnout = candidate_burnout
                    best_sleep = test_sleep
                    best_study = test_study
                    best_plan = {
                        'recommended_sleep': float(best_sleep),
                        'recommended_study': float(best_study),
                        'new_predicted_score': float(best_score),
                        'new_burnout_probability': float(best_burnout),
                        'score_improvement': float(best_score - current_score),
                        'burnout_change': float(best_burnout - current_burnout)
                    }

    if best_plan is None:
        # If no positive improvement found under constraints
        best_plan = {
            'recommended_sleep': float(current_sleep),
            'recommended_study': float(current_study),
            'new_predicted_score': float(current_score),
            'new_burnout_probability': float(current_burnout),
            'score_improvement': 0.0,
            'burnout_change': 0.0
        }
        
    return best_plan
