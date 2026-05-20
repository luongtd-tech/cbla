import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_metrics(y_true, y_pred):
    """
    Calculates RMSE, MAE, and R2 as specified in the paper.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def print_evaluation(metrics_dict, model_name="Model"):
    print(f"--- Evaluation for {model_name} ---")
    print(f"RMSE: {metrics_dict['RMSE']:.4f}")
    print(f"MAE:  {metrics_dict['MAE']:.4f}")
    print(f"R2:   {metrics_dict['R2']:.4f}")
    print("-" * 35)
