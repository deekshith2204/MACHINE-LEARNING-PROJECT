import os
import sys

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerconfig:
    model_trained_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting training and testing data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            
            models={
                'linear model':LinearRegression(),
                'random forest regressor':RandomForestRegressor(),
                'xgboost':XGBRegressor(),
                'decision tree':DecisionTreeRegressor(),
                'kneiborsregreesor':KNeighborsRegressor(),
                'adaboost':AdaBoostRegressor(),
                'gradientboost':GradientBoostingRegressor()
                }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_Score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_Score)
            ]
            best_model=models[best_model_name]
            if best_model_Score<0.6:
                raise CustomException("no best model found")
            logging.info("best model on both training and testing")
            save_object(
                file_path=self.model_trainer_config.model_trained_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)

            r2scr=r2_score(y_test,predicted)
            return r2scr

        except Exception as e:
            raise CustomException(e,sys)


