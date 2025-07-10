import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_configs import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        
        self.config = read_yaml(config_path)
        

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

        logger.info(f"Data Processing initialized with config: {self.config}")

    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing...")

            logger.info("Dropping Columns and Duplicates")
            df.drop(['Booking_ID'], inplace=True, axis=1)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_features"]
            num_cols = self.config["data_processing"]["numerical_features"]

            logger.info("Label Encoding Categorical Features")
            le = LabelEncoder()
            mapping = {}

            for col in cat_cols:
                if df[col].dtype == 'object':
                    df[col] = le.fit_transform(df[col])
                    mapping[col] = {label: code for label, code in zip(le.classes_, le.transform(le.classes_))}

            logger.info("Label Encoding Completed")
            for col_name, label_map in mapping.items():
                logger.info(f"Mapping for {col_name}: {label_map}")

            logger.info("Handling Data Skewness")
            threshold = self.config["data_processing"]["skewness_threshhold"]
            skewness = df[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness > threshold].index:
                logger.info(f"Applying log transformation to {col} (skewness = {skewness[col]})")
                df[col] = np.log1p(df[col])

            logger.info("DataFrame preprocessing complete")
            return df

        except Exception as e:
            logger.error(f"Error in data preprocessing:", e)
            

    def balance_data(self, df):
        try:
            logger.info("Handling class imbalance using SMOTE...")
            X = df.drop(["booking_status"], axis=1)
            y = df["booking_status"]

            ros = SMOTE(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)

            balanced_df = pd.concat([X_resampled, y_resampled], axis=1)

            logger.info("Class imbalance handled successfully.")
            return balanced_df

        except Exception as e:
            logger.error(f"Error in balancing data: {e}")
            raise CustomException(f"Error in balancing data:", e)

    def select_features(self, df):
        try:
            logger.info("Selecting top 10 important features using RandomForestClassifier...")
            X = df.drop(["booking_status"], axis=1)
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance
            })

            top_10_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)
            selected_features = top_10_features["Feature"].tolist()

            logger.info(f"Top 10 features selected: {selected_features}")
            return selected_features

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException("Error in feature selection:" ,e)

    def save_data(self, df, file_path):
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise CustomException("Error saving processed data:" ,e)

    def process(self):
        try:
            logger.info("Starting full data processing pipeline...")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            selected_features = self.select_features(train_df)


            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("✅ Data processing pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            raise CustomException(f"Error in data preprocessing:" , e)


if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
