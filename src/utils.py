import os 
import sys 
import numpy as np
import pandas as pd 
import dill 
from src.exception import CustomException

# General function ton save any class object initilized
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)