import os 
import sys 
import numpy as np
import pandas as pd 
import dill 
from src.exception import CustomException
from sklearn.metrics import r2_score

# General function ton save any class object initilized
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        # for i in range(len(list(models))):
        for model_name, model in models.items():
            # model = list(models.values())[i]
            model.fit(X_train, y_train)            # Training the model on training set
            
            y_train_pred = model.predict(X_train)  # Making predictions on training and testing sets
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            #report[list(model.keys())[i]] = test_model_score
            report[model_name] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e, sys)