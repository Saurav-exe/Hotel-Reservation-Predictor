import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from src.custom_exception import CustomException
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score
from src.logger import get_logger
from config.paths_configs import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint, uniform

import mlflow
import mlflow.sklearn


logger= get_logger(__name__)

class ModelTraining:

  def __init__(self,train_path, test_path, model_output_path):

    self.train_path= train_path #train data path
    self.test_path= test_path #test data path
    self.model_output_path= model_output_path

    self.params_dist=LGBM_PARAMS
    self.random_search_params= RANDOM_SEARCH_PARAMS

  def load_and_split_data(self):
    try:
      logger.info(f"Loading data from {self.train_path}" )
      train_df= load_data(self.train_path)

      logger.info(f"Loading data from {self.test_path}" )
      test_df= load_data(self.test_path)

      X_train = train_df.drop(['booking_status'], axis=1)

      y_train = train_df['booking_status']

      X_test = test_df.drop(['booking_status'],axis=1)
      y_test = test_df['booking_status']

      logger.info("Data loaded and split into features and target variable.")

      return X_train, y_train, X_test, y_test
    
    except Exception as e:
      logger.error(f"Error in loading and splitting data: {str(e)}")
      raise CustomException("Error in loading and splitting data:",e)

  def train_lgbm(self, X_train, y_train):

    try:
      logger.info("Starting LightGBM model training...")

      lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

      logger.info("Performing Randomized Search for hyperparameter tuning...")

      random_search = RandomizedSearchCV(
          estimator=lgbm_model,
          param_distributions=self.params_dist,
          n_iter=self.random_search_params['n_iter'],
          cv=self.random_search_params['cv'],
          n_jobs=self.random_search_params['n_jobs'],
          verbose=self.random_search_params['verbose'],
          random_state=self.random_search_params['random_state'],
          scoring=self.random_search_params['scoring']
      )

      logger.info("Fitting the model with training data...")

      random_search.fit(X_train, y_train)

      
      logger.info("Training completed successfully.")

      best_params= random_search.best_params_
      best_lgbm_model= random_search.best_estimator_

      logger.info(f"Best parameters found: {random_search.best_params_}")

      return best_lgbm_model

    except Exception as e:
      logger.error(f"Error in training LightGBM model: {str(e)}")
      raise CustomException("Error in training LightGBM model:", e)
    

  def evaluate_model(self, model, X_test, y_test):
    try:
      logger.info("Evaluating the trained model...")

      y_pred=  model.predict(X_test)
      accuracy= accuracy_score(y_test, y_pred)
      f1= f1_score(y_test, y_pred, average='weighted')
      recall= recall_score(y_test, y_pred, average='weighted')
      precision= precision_score(y_test, y_pred, average='weighted')

      logger.info(f"Model evaluation results - \n Accuracy: {accuracy} \n F1 Score: {f1} \n Recall: {recall}\n Precision: {precision}")

      return {
          'accuracy': accuracy,
          'f1_score': f1,
          'recall': recall,
          'precision': precision
      }
    except Exception as e:
      logger.error(f"Error in evaluating model: {str(e)}")
      raise CustomException("Error in evaluating model:", e)
    

  def save_model(self, model):
    try:
      os.makedirs(self.model_output_path, exist_ok=True)
      logger.info(f"Saving the model to {self.model_output_path} ")

      joblib.dump(model, os.path.join(self.model_output_path, 'lgbm_model.pkl'))

    except Exception as e:
      logger.error(f"Error in saving model: {str(e)}")
      raise CustomException("Failed to save model:", e) 


  def run(self):
    try:

      with mlflow.start_run():

        logger.info("Starting the model training process with MLFLOW experimentation...")

        logger.info("Logging train and test data to MLflow")

        mlflow.log_artifact(self.train_path, artifact_path="datasets")
        mlflow.log_artifact(self.test_path, artifact_path="datasets")

        X_train, y_train, X_test, y_test = self.load_and_split_data()
        best_lgbm_model= self.train_lgbm(X_train, y_train)
        metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
        self.save_model(best_lgbm_model)

        logger.info("Logging model & params to MLflow")
        mlflow.log_artifact(self.model_output_path, artifact_path="models")
        mlflow.log_params(best_lgbm_model.get_params())
        mlflow.log_metrics(metrics)

        logger.info("Model training process completed successfully.")


    except Exception as e:  
      logger.error(f"Error in Model Training: {str(e)}")
      raise CustomException("Failed to Train model:", e) 


if __name__ == "__main__":
  trainer = ModelTraining(
    train_path= PROCESSED_TRAIN_DATA_PATH, 
    test_path= PROCESSED_TEST_DATA_PATH, 
    model_output_path=MODEL_OUTPUT_PATH)
  trainer.run()

