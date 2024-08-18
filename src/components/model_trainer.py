import os
import sys
import warnings
import numpy as np 
import matplotlib.pyplot as plt 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression" : LinearRegression(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                #"CatBoosting Regressor": CatBoostRegressor(),
                "XGBRegressor" : XGBRegressor()
            }
            
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train,X_test = X_test,y_test = y_test, models = models)
            # Get the best model from dict (one with highest score)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name] 
            # Setting a performance threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Training & evaluation of several models completed")
            
            save_object(file_path = self.model_trainer_config.train_model_file_path, obj = best_model)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return best_model,r2_square
        except Exception as e: 
            raise CustomException(e, sys)