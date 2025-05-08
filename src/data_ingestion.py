import os 
import logging 
import pandas as pd 
from data_io import load_data, save_data


# Ensure "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting logging configuration
logger = logging.getLogger(name="data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"data_ingestion.log")
file_handler = logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def dataframe_restructuring(df: pd.DataFrame) -> pd.DataFrame:
    """"
    Restructres Dataframe by dropping irrelevant features and renaming relevant ones
    """    
    try:
        feature_rename_dict = {
        "Задолженность" : "DEBT",
        "Просрочка, дни" : "OVERDUE DAYS",
        "Первоначльный лимит" : "INITIAL LIMIT",
        "Рейтинг кредитной истории" : "CREDIT HISTORY RATING",
            }
        df.rename(columns=feature_rename_dict, inplace=True)
        df.drop('CLIENTID', axis=1, inplace=True)
        df.drop(df[df["PDN"] > 1].index,axis=0,inplace=True)
        logger.debug("Restructured DataFrame Successfully")
    except KeyError as e:
        logger.error(f"Columns intended to go through changes are missing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while restructuring DataFrame: {e}")
        raise

def main() -> None:
    try:
        df = load_data(data_path="./data", filename="bank_credit_scoring.csv",logger=logger)
        final_df = dataframe_restructuring(df=df)
        save_data(df=df,save_data_path='./raw_data',filename="raw.csv",logger=logger)
        logger.debug(f"Data Ingestion Completed Successfully")
    except Exception as e:
        logger.error(f"Data Ingestion Failed: {e}")
        raise

if __name__ == "__main__":
    main()
