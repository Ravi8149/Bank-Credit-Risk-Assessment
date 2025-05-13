import pandas as pd
import os
import yaml

def load_params(filepath: str, logger) -> dict:
    """
    Loads params from a given yaml file to a python dictionary
    """
    try:
        with open(filepath,'r') as file:
            params = yaml.safe_load(file)
        return params
        logger.debug(f"Parameters retrieved from: {filepath}")
    except FileNotFoundError as e:
        logger.error(f"File not Found: {e}")
    except yaml.YAMLError as e:
        logger.error(f"YAML Error: {e}")
    except Exception as e:
        logger.error("Unexpected error occured while loading params from yaml: {e}")
    

def load_data(data_path: str, filename: str, logger) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas Dataframe and returns it
    """
    try:
        filepath = os.path.join(data_path,filename)
        df = pd.read_csv(filepath)
        logger.debug(f"Data Loaded from {filepath} successfully")
        return df
    except FileNotFoundError as e:
        logger.error(f"FileNotFound at provided data_path: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Pandas coundn't parse the CSV into dataframe: {e}")
        raise


def save_data(df: pd.DataFrame, save_data_path: str, filename: str, logger) -> None:
    """
    Saves the given pandas DataFrame as CSV to provided saving path
    """
    try:
        os.makedirs(save_data_path, exist_ok=True)
        data_save_path = os.path.join(save_data_path,filename)
        df.to_csv(data_save_path, index=False)
        logger.debug(f"Data saved successfully at {data_save_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving data: {e}")
        raise