import os 
import logging 
from data_io import load_data, save_data, load_params
import pandas as pd
# from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import pickle

# Ensure "logs" dir
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting logging configuration
logger = logging.getLogger(name="model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"model_building.log")
file_handler = logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def model_training(X_train, y_train, params: dict):
    """
    Trains a regression model for given data
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            logger.error("The number of samples in X_train and y_train must be the same.")
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug("Initializing Model")
        regressor = LinearRegression(fit_intercept=params["fit_intercept"],
                                     copy_X=params["copy_X"],
                                     n_jobs=params["n_jobs"],
                                     positive=params["positive"])
        regressor.fit(X_train,y_train)
        logger.debug(f"Model Training Successful")
        return regressor
    except Exception as e:
        logger.error(f"Unexpected error occured while Model Builidng: {e}")
        raise

def save_model(model, model_save_path: str, filename: str) -> None:
    """
    Saves the model into a pickle file format
    """
    try:
        model_save_path = os.path.join(model_save_path,filename)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved successfully")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving model: {e}")
        raise 
    
    
def main() -> None:
    try:
        params = load_params(filepath="./params.yaml",logger=logger)
        model_building_params = params["model_building"]
        df = load_data(data_path="./data/feature_engineered_data/",filename="X_train_feature_engineered.csv",logger=logger)
        X_train = df.drop("PDN",axis=1)
        y_train = df["PDN"]
        model = model_training(X_train=X_train,y_train=y_train,params=model_building_params)
        save_model(model=model,model_save_path="./models/",filename="model.pkl")
        logger.debug(f"Model building completed successfully")
    except Exception as e:
        logger.error(f"Model Building Failed: {e}")
        raise
        

if __name__ == "__main__":
    main()