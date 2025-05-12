import os 
import logging
import pandas as pd
from data_io import load_data, save_data
from translations import translation_dict
# from deep_translator import GoogleTranslator


# Ensure "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting logging configurations
logger = logging.getLogger(name="data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"data_preprocessing.log")
file_handler = logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# def russian_to_english(df: pd.DataFrame, column_name: str) -> dict:
#     try:
#         translator = GoogleTranslator(source='ru', target='en')
#         keys = df[column_name].unique()
#         pairs = [translator.translate(text) for text in keys]
#         translation_mapping = dict(zip(keys,pairs))
#         logger.debug(f"Translated data from column {column_name} successfully")
#         translation_df = pd.DataFrame({"original_text":keys,"translation":pairs})
#         translation_save_path = f"./translations"
#         os.makedirs(translation_save_path, exist_ok=True)
#         translation_df.to_csv(os.path.join(translation_save_path,f"{column_name}.csv"), index=False)
#         logger.debug(f"Translation of {column_name} saved to {translation_save_path}")
#         return translation_mapping
#     except KeyError as e:
#         logger.error(f"Given column might not exist in provided pandas DataFrame: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error occured while translation: {e}")


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate records and translates russian text to english
    """
    try:
        # remove duplicate values
        duplicate_count = df.duplicated().sum()
        if duplicate_count >0:
            df.drop_duplicates(keep='first', inplace=True)
            logger.debug(f"{duplicate_count} duplicate records droped successfully")
        # translation from russian to english
        # for column in translation_columns:
        #     df[column] = df[column].map(russian_to_english(df=df,column_name=column))

        # sex column translation
        sex_translation = translation_dict["sex_translation"]
        df["SEX"] = df["SEX"].map(sex_translation)
        logger.debug("SEX column translated successfully")


        # edu column translation
        edu_translation = translation_dict["edu_translation"]
        df["EDU"] = df["EDU"].map(edu_translation)
        logger.debug("EDU column translated successfully")


        # industry column translation
        industry_translation = translation_dict["industry_translation"]
        df["INDUSTRYNAME"] = df["INDUSTRYNAME"].map(industry_translation)
        logger.debug("INDUSTRYNAME column translated successfully")

        # lv area translation
        lv_area_translation = translation_dict["lv_area_translation"]
        df["LV_AREA"] = df["LV_AREA"].map(lv_area_translation)
        logger.debug("LV_AREA column translated successfully")
        return df
    except Exception as e:
        logger.error(f"Unexpected error occured while data preprocessing: {e}")
        raise


def main():
    try:
        input_filepath = './data/raw_data'
        df = load_data(data_path=input_filepath,filename="raw.csv",logger=logger)
        # translation_columns = ["SEX","EDU","LV_AREA","INDUSTRYNAME"]
        processed_df = data_preprocessing(df=df)
        save_data(df=processed_df, save_data_path="./data/processed_data/",filename="processed.csv",logger=logger)
    except Exception as e:
        logger.error(f"Data Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()