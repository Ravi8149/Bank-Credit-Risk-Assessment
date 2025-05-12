import os 
import logging
import pandas as pd
from data_io import load_data, save_data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting up logging configurations
logger = logging.getLogger(name="feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"feature_engineering.log")
file_handler = logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def feature_handling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles features like BIRTHDATE and gets AGE out of it and drops the original BIRTHDATE feature
    """
    try:
        df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"])
        df["AGE"] = (pd.Timestamp.today().year - df["BIRTHDATE"].dt.year)
        df.drop("BIRTHDATE",axis=1,inplace=True)
        return df
    except Exception as e:
        logger.error(f"Unexpected error occured while creating AGE feature: {e}")
        raise



def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consists of imputation and encoding
    """
    try:
        edu_ordinal_encoding = [
            'Secondary',
            'Secondary specialized',
            'Incomplete higher education',
            'Higher education',
            'Postgraduate'
        ]
        credit_rating_ordinal_encoding = [
            'E3', 'E2', 'E1', 'D3', 'D2', 'D1', 'C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1', '-1'
        ]

        imputer_transformer = ColumnTransformer(
            [
                ("Mode Imputation",SimpleImputer(strategy="most_frequent"),["CREDIT HISTORY RATING","LV_AREA","VELCOMSCORING"]),
                ("Mean Imputation",SimpleImputer(strategy="mean"),["SCORINGMARK"])
            ], remainder="passthrough",
            verbose_feature_names_out=False
        )
        imputer_transformer.set_output(transform="pandas")

        encoding_transformer = ColumnTransformer(
            [
                ("Ordinal Encoding",OrdinalEncoder(categories=[edu_ordinal_encoding, credit_rating_ordinal_encoding]),["EDU","CREDIT HISTORY RATING"]),
                ("OneHotEncoding",OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),["SEX","LV_AREA","INDUSTRYNAME"])
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )
        encoding_transformer.set_output(transform="pandas")
        
        pipeline = Pipeline(
            [
                ("imputation",imputer_transformer),
                ("encoding",encoding_transformer)
            ]
        )         
        y_true = df["PDN"].values 
        df.drop("PDN",axis=1,inplace=True)
        feature_engineered_df = pipeline.fit_transform(df)
        std = StandardScaler()
        standardized_df = std.fit_transform(feature_engineered_df)
        feature_engineered_df = pd.DataFrame(standardized_df,columns=feature_engineered_df.columns)
        feature_engineered_df["PDN"] = y_true
        logger.debug(f"Feature Engineering Completed")
        return feature_engineered_df
    except Exception as e:
        logger.error(f"Expected error occured while feature engineering: {e}")
        raise


def main() -> None:
    try:
        df = load_data(data_path="./data/processed_data/",filename="processed.csv",logger=logger)
        df = feature_handling(df=df)
        X_train, X_test = train_test_split(df, test_size=0.2)
        logger.debug(f"Feature Engineering for Training Data Started")
        feature_engineered_X_train = feature_engineering(df=X_train)
        logger.debug(f"Feature Engineering for Test Data Started")
        feature_engineered_X_test = feature_engineering(df=X_test)
        save_data(df=feature_engineered_X_train,save_data_path="./data/feature_engineered_data/",filename="X_train_feature_engineered.csv",logger=logger)
        save_data(df=feature_engineered_X_test,save_data_path="./data/feature_engineered_data/",filename="X_test_feature_engineered.csv",logger=logger)
        logger.debug(f"Feature Engineering completed successfully")
    except Exception as e:
        logger.error(f"Feature Engineering: {e}")
        raise


if __name__ == "__main__":
    main()