import os 
import logging 
from data_io import load_data, save_data
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# Ensure "logs" dir
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting logging configuration
logger = logging.getLogger(name="model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"model_evaluation.log")
file_handler = logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(model_file_path: str):
    """
    Loads model from a pickle file
    """
    try:
        with open(model_file_path,"rb") as file:
            model = pickle.load(file=file)
        logger.debug('Model loaded from %s', model_file_path)
        return model
    except FileNotFoundError as e:
        logger.error(f"File not found at {model_file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while loading model: {e}")

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the model and return the evaluation metrics
    """
    try: 
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        n, p = X_test.shape
        r2_adj = 1 - (((1 - r2) * (n - 1)) / (n - p - 1))

        metric_dict = {
            "mse":mse,
            "rmse":rmse,
            "mae":mae,
            "r2":r2,
            "r2_adj":r2_adj
        }
        logger.debug(f"Model evaluation metrics calculated")
        return metric_dict
    except Exception as e:
        logger.error(f"Unexpected error occured while calculating evaluation metrics: {e}")
        raise

def save_metrics(metrics: dict, save_file_path: str) -> None:
    """
    Saves evaluation metrics as a JSON at specified file path
    """
    try:
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        with open(save_file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Evaluation metrics saved successfully to {save_file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving metrics: {e}")
        raise


def main() -> None:
    try:
        df = load_data(data_path="./feature_engineered_data",filename="X_test_feature_engineered.csv",logger=logger)
        X_test = df.drop("PDN",axis=1)
        y_true = df["PDN"]

        model = load_model(model_file_path="./models/model.pkl")
        metrics = evaluate_model(model=model,X_test=X_test,y_test=y_true)
        save_metrics(metrics=metrics,save_file_path="./reports/metrics.json")
        logger.debug("Model Evaluation Completed Successfully")
    except Exception as e:
        logger.error("Model Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()